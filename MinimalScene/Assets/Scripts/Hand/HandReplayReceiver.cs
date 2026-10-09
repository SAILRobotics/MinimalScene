using UnityEngine;
using UnityEngine.Rendering;
using NetMQ;
using NetMQ.Sockets;
using System;
using System.Threading;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Newtonsoft.Json;

using Oculus.Interaction;
using Oculus.Interaction.Input;

/// <summary>
/// Receives hand recordings from hand_record_replay.py (L / R keys) and plays each one
/// back once, then hides it. Left and right play independently.
///
/// The replay always shows the hand mesh — a tinted copy of the scene's HandVisual model,
/// posed exactly the way HandVisual poses it (root pose + scale + local joint poses).
/// The joint/bone skeleton is drawn on top only when the replay asks for it
/// (python main_hands_simple.py --record-replay --replay-skeleton), or when a recording
/// has no mesh data.
///
/// Also tints the live hand markers (TrackingDataVisualizer) while Python is recording
/// that hand.
///
/// Recordings are stored relative to WorldRoot (ArUco marker 100), so put this on
/// WorldRoot — or set replayRoot to it — and the ghost replays on the same physical spot.
///
/// Messages (JSON, SUB bound on tcp://0.0.0.0:port):
///   {"type": "replay", "hand": "left"|"right", "skeleton": bool,
///    "frames": [{"t": s, "p": [22 * xyz], "r": [xyz, xyzw], "s": scale, "j": [N * (xyz, xyzw)]}]}
///   {"type": "record_state", "hand": "left"|"right", "recording": bool, "color": [r, g, b]}
/// </summary>
public class HandReplayReceiver : MonoBehaviour
{
    [Header("Targets")]
    [Tooltip("Parent for the replay ghosts — WorldRoot. Defaults to this transform.")]
    [SerializeField] private Transform replayRoot;
    [Tooltip("Live hand markers to tint while recording. Found automatically if empty.")]
    [SerializeField] private TrackingDataVisualizer liveVisualizer;

    [Header("Hand mesh (always shown)")]
    [Tooltip("Hand visuals whose model is copied for the replay mesh. Found automatically if empty.")]
    [SerializeField] private HandVisual leftHandVisual;
    [SerializeField] private HandVisual rightHandVisual;
    [Tooltip("Tint the replay mesh with the hand colour below so it can't be mistaken for the live hand.")]
    [SerializeField] private bool tintMesh = true;

    [Header("Skeleton (joints + bones, toggled by Python's --replay-skeleton)")]
    [Tooltip("Joint marker prefab, e.g. Assets/Scripts/Hand/FingerJoint.")]
    [SerializeField] private GameObject jointPrefab;
    [SerializeField] private float boneWidth = 0.004f;

    [Header("Colours")]
    [SerializeField] private Color leftColor  = new Color(0.3f, 0.6f, 1.0f, 1f);
    [SerializeField] private Color rightColor = new Color(1.0f, 0.55f, 0.1f, 1f);

    [Header("NetMQ")]
    [SerializeField] private int port = 6003;

    [Header("Debug")]
    [SerializeField] private bool log = true;

    private class ReplayFrame
    {
        public float t;
        public float[] p;    // 22 joint positions (skeleton)
        public float[] r;    // mesh root pose: position xyz + rotation xyzw (WorldRoot-local)
        public float s = 1f; // mesh scale
        public float[] j;    // mesh joints, local to parent bone: N * (xyz + xyzw)
    }

    private class Message
    {
        public string type;
        public string hand;
        public bool recording;
        public bool skeleton;
        public float[] color;
        public List<ReplayFrame> frames;
    }

    private static readonly int BaseColorID   = Shader.PropertyToID("_BaseColor");
    private static readonly int ColorID       = Shader.PropertyToID("_Color");
    private static readonly int ColorTopID    = Shader.PropertyToID("_ColorTop");     // Interaction/OculusHand
    private static readonly int ColorBottomID = Shader.PropertyToID("_ColorBottom");

    private Thread receiveThread;
    private volatile bool isRunning = false;
    private bool registered = false;
    private bool hasShutdown = false;
    private readonly ConcurrentQueue<Message> inbox = new();

    private Ghost leftGhost;
    private Ghost rightGhost;
    private readonly Dictionary<string, bool> tinted = new() { { "left", false }, { "right", false } };

    void Start()
    {
        if (replayRoot == null)
            replayRoot = transform;
        if (liveVisualizer == null)
            liveVisualizer = FindFirstObjectByType<TrackingDataVisualizer>();
        FindHandVisuals();
        if (jointPrefab == null)
            Debug.LogWarning("[HandReplay] jointPrefab is not assigned — skeleton uses plain spheres.", this);

        leftGhost  = new Ghost("ReplayLeftHand",  replayRoot, jointPrefab, leftHandVisual,
                               leftColor,  tintMesh, boneWidth);
        rightGhost = new Ghost("ReplayRightHand", replayRoot, jointPrefab, rightHandVisual,
                               rightColor, tintMesh, boneWidth);

        NetMQManager.RegisterReceiver();
        registered = true;
        isRunning = true;
        receiveThread = new Thread(ReceiveLoop) { IsBackground = true };
        receiveThread.Start();

        if (log) Debug.Log($"[HandReplay] 📡 Listening on tcp://0.0.0.0:{port}");
    }

    private void FindHandVisuals()
    {
        if (leftHandVisual != null && rightHandVisual != null)
            return;

        foreach (var visual in FindObjectsByType<HandVisual>(FindObjectsInactive.Include, FindObjectsSortMode.None))
        {
            bool isLeft = visual.Hand != null
                ? visual.Hand.Handedness == Handedness.Left
                : visual.name.IndexOf("Left", StringComparison.OrdinalIgnoreCase) >= 0;
            if (isLeft && leftHandVisual == null)
                leftHandVisual = visual;
            else if (!isLeft && rightHandVisual == null)
                rightHandVisual = visual;
        }

        if (leftHandVisual == null || rightHandVisual == null)
            Debug.LogWarning("[HandReplay] No HandVisual found for one or both hands — " +
                             "those replays fall back to the skeleton.", this);
    }

    // The socket lives entirely on this thread (created, used and disposed here), so
    // shutdown only has to stop the loop and join — no cross-thread socket access.
    private void ReceiveLoop()
    {
        AsyncIO.ForceDotNet.Force();

        try
        {
            using (var subscriber = new SubscriberSocket())
            {
                subscriber.Bind($"tcp://0.0.0.0:{port}");
                subscriber.Subscribe("");

                while (isRunning)
                {
                    if (!subscriber.TryReceiveFrameString(TimeSpan.FromMilliseconds(100), out string json))
                        continue;
                    try
                    {
                        // Replays can be large; parse here rather than on the main thread.
                        var msg = JsonConvert.DeserializeObject<Message>(json);
                        if (msg != null)
                            inbox.Enqueue(msg);
                    }
                    catch (Exception e)
                    {
                        Debug.LogWarning("[HandReplay] ⚠️ Bad message: " + e.Message);
                    }
                }
            }
        }
        catch (TerminatingException) { }
        catch (Exception e)
        {
            if (isRunning)
                Debug.LogWarning("[HandReplay] ❌ ReceiveLoop exception: " + e.Message);
        }
    }

    void Update()
    {
        while (inbox.TryDequeue(out var msg))
        {
            if (msg.hand != "left" && msg.hand != "right")
                continue;

            if (msg.type == "replay")
            {
                var ghost = msg.hand == "left" ? leftGhost : rightGhost;
                if (ghost.Play(msg.frames, msg.skeleton) && log)
                    Debug.Log($"[HandReplay] ▶️ {msg.hand} hand: {msg.frames.Count} frames, " +
                              $"{msg.frames[msg.frames.Count - 1].t:F1} s, " +
                              $"mesh={ghost.ShowingMesh}, skeleton={ghost.ShowingSkeleton}");
            }
            else if (msg.type == "record_state" && tinted[msg.hand] != msg.recording)
            {
                tinted[msg.hand] = msg.recording;
                Color? tint = null;
                if (msg.recording)
                    tint = msg.color != null && msg.color.Length >= 3
                        ? new Color(msg.color[0], msg.color[1], msg.color[2], 1f)
                        : Color.red;
                if (liveVisualizer != null)
                    liveVisualizer.SetHandTint(msg.hand == "left" ? "Left" : "Right", tint);
                if (log) Debug.Log($"[HandReplay] {msg.hand} hand recording: {msg.recording}");
            }
        }

        leftGhost?.Tick();
        rightGhost?.Tick();

        if (NetMQManager.IsShutdownRequested)
            Shutdown();
    }

    /// <summary>One hand's replay: always the mesh, optionally the skeleton; played once then hidden.</summary>
    private class Ghost
    {
        private readonly Transform replayRoot;
        private readonly SkeletonGhost skeleton;
        private readonly MeshGhost mesh;   // null if no HandVisual to copy
        private List<ReplayFrame> frames;
        private float startTime;
        private int cursor;

        public bool ShowingMesh { get; private set; }
        public bool ShowingSkeleton { get; private set; }

        public Ghost(string name, Transform replayRoot, GameObject jointPrefab, HandVisual handVisual,
                     Color color, bool tintMesh, float boneWidth)
        {
            this.replayRoot = replayRoot;
            skeleton = new SkeletonGhost(name + "Skeleton", replayRoot, jointPrefab, color, boneWidth);
            if (handVisual != null)
                mesh = new MeshGhost(name + "Mesh", replayRoot, handVisual, tintMesh ? color : (Color?)null);
        }

        public bool Play(List<ReplayFrame> newFrames, bool showSkeleton)
        {
            if (newFrames == null || newFrames.Count == 0)
                return false;
            frames = newFrames;
            startTime = Time.time;
            cursor = 0;

            // Always show the mesh; fall back to the skeleton if this recording (or scene) has no mesh.
            ShowingMesh = mesh != null && MeshGhost.HasData(frames[0]);
            ShowingSkeleton = showSkeleton || !ShowingMesh;

            Apply(frames[0], frames[0], 0f);
            mesh?.SetVisible(ShowingMesh);
            skeleton.SetVisible(ShowingSkeleton);
            return true;
        }

        public void Tick()
        {
            if (frames == null)
                return;

            float t = Time.time - startTime;
            if (t > frames[frames.Count - 1].t)
            {
                mesh?.SetVisible(false);   // played once — disappear
                skeleton.SetVisible(false);
                frames = null;
                return;
            }

            while (cursor + 1 < frames.Count && frames[cursor + 1].t <= t)
                cursor++;
            var a = frames[cursor];
            var b = frames[Mathf.Min(cursor + 1, frames.Count - 1)];
            float span = b.t - a.t;
            Apply(a, b, span > 1e-5f ? (t - a.t) / span : 0f);
        }

        private void Apply(ReplayFrame a, ReplayFrame b, float alpha)
        {
            if (ShowingMesh)
                mesh.Apply(a, b, alpha, replayRoot);
            if (ShowingSkeleton)
                skeleton.Apply(a.p, b.p, alpha);
        }
    }

    /// <summary>Joint markers + bone lines for the 22 tracked joints.</summary>
    private class SkeletonGhost
    {
        private readonly GameObject root;
        private readonly Transform[] joints = new Transform[TrackingUtils.JOINT_COUNT];
        private readonly LineRenderer[] bones;

        public SkeletonGhost(string name, Transform parent, GameObject jointPrefab, Color color, float boneWidth)
        {
            root = new GameObject(name);
            root.transform.SetParent(parent, false);

            var block = new MaterialPropertyBlock();
            block.SetColor(BaseColorID, color);
            block.SetColor(ColorID, color);

            Material boneMaterial = null;
            for (int j = 0; j < joints.Length; j++)
            {
                GameObject marker;
                if (jointPrefab != null)
                {
                    marker = UnityEngine.Object.Instantiate(jointPrefab, root.transform);
                }
                else
                {
                    marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    marker.transform.SetParent(root.transform, false);
                    marker.transform.localScale = Vector3.one * 0.01f;
                }
                marker.name = $"Joint{j}";
                // Ghosts are visual only — keep them out of physics / ray interaction.
                foreach (var collider in marker.GetComponentsInChildren<Collider>(true))
                    UnityEngine.Object.Destroy(collider);
                foreach (var renderer in marker.GetComponentsInChildren<Renderer>(true))
                {
                    renderer.SetPropertyBlock(block);
                    boneMaterial ??= renderer.sharedMaterial;
                }
                joints[j] = marker.transform;
            }

            bones = new LineRenderer[TrackingUtils.HAND_BONES.GetLength(0)];
            for (int b = 0; b < bones.Length; b++)
            {
                var line = new GameObject($"Bone{b}").AddComponent<LineRenderer>();
                line.transform.SetParent(root.transform, false);
                line.useWorldSpace = false;
                line.positionCount = 2;
                line.widthMultiplier = boneWidth;
                line.shadowCastingMode = ShadowCastingMode.Off;
                line.receiveShadows = false;
                if (boneMaterial != null)
                    line.sharedMaterial = boneMaterial;
                line.SetPropertyBlock(block);
                bones[b] = line;
            }

            root.SetActive(false);
        }

        public void SetVisible(bool visible) => root.SetActive(visible);

        public void Apply(float[] from, float[] to, float alpha)
        {
            if (from == null || to == null || from.Length < joints.Length * 3 || to.Length < joints.Length * 3)
                return;

            for (int j = 0; j < joints.Length; j++)
            {
                int i = j * 3;
                joints[j].localPosition = Vector3.Lerp(new Vector3(from[i], from[i + 1], from[i + 2]),
                                                       new Vector3(to[i], to[i + 1], to[i + 2]), alpha);
            }
            for (int b = 0; b < bones.Length; b++)
            {
                bones[b].SetPosition(0, joints[TrackingUtils.HAND_BONES[b, 0]].localPosition);
                bones[b].SetPosition(1, joints[TrackingUtils.HAND_BONES[b, 1]].localPosition);
            }
        }
    }

    /// <summary>
    /// A script-free copy of a HandVisual's model, posed the way HandVisual.UpdateSkeleton()
    /// poses it: Root gets the hand root pose and scale, each joint its parent-relative pose.
    /// </summary>
    private class MeshGhost
    {
        private readonly GameObject container;
        private readonly Transform root;
        private readonly Transform[] joints;   // HandJointId order, like HandVisual.Joints
        private readonly Renderer[] renderers;

        public static bool HasData(ReplayFrame f) =>
            f.r != null && f.r.Length >= 7 && f.j != null && f.j.Length >= 7;

        public MeshGhost(string name, Transform parent, HandVisual source, Color? tint)
        {
            // Copy under an inactive parent so none of the copied scripts ever wake up
            // (a copied HandVisual would otherwise keep following the live hand).
            container = new GameObject(name);
            container.SetActive(false);
            container.transform.SetParent(parent, false);

            var copy = UnityEngine.Object.Instantiate(source.gameObject, container.transform);
            copy.name = source.name + " (replay)";
            copy.transform.SetLocalPositionAndRotation(Vector3.zero, Quaternion.identity);
            copy.transform.localScale = Vector3.one;
            copy.SetActive(true);   // still inactive in the hierarchy until SetVisible(true)
            StripScripts(copy);

            root = Counterpart(source.Root, source.transform, copy.transform) ?? copy.transform;
            joints = new Transform[source.Joints.Count];
            for (int i = 0; i < joints.Length; i++)
                joints[i] = Counterpart(source.Joints[i], source.transform, copy.transform);
            renderers = copy.GetComponentsInChildren<Renderer>(true);

            if (tint.HasValue)
            {
                var block = new MaterialPropertyBlock();
                Color top = tint.Value;
                Color bottom = Color.Lerp(top, Color.black, 0.35f);
                block.SetColor(ColorTopID, top);
                block.SetColor(ColorBottomID, bottom);
                block.SetColor(BaseColorID, top);
                block.SetColor(ColorID, top);
                foreach (var renderer in renderers)
                    renderer.SetPropertyBlock(block);
            }

            if (joints.Length == 0 || Array.TrueForAll(joints, j => j == null))
                Debug.LogWarning($"[HandReplay] Could not map {source.name}'s joints onto the replay copy.");
        }

        public void SetVisible(bool visible)
        {
            // HandVisual disables its SkinnedMeshRenderer while the live hand is untracked,
            // so the copy may have inherited a disabled renderer — force it back on.
            if (visible)
                foreach (var renderer in renderers)
                    renderer.enabled = true;
            container.SetActive(visible);
        }

        public void Apply(ReplayFrame a, ReplayFrame b, float alpha, Transform replayRoot)
        {
            if (!HasData(a) || !HasData(b))
                return;

            Pose rootPose = LerpPose(a.r, b.r, 0, alpha);
            root.SetPositionAndRotation(replayRoot.TransformPoint(rootPose.position),
                                        replayRoot.rotation * rootPose.rotation);
            float scale = Mathf.Lerp(a.s, b.s, alpha);
            float parentScale = root.parent != null ? root.parent.lossyScale.x : 1f;
            root.localScale = (scale > 0f ? scale : 1f) / parentScale * Vector3.one;

            int count = Mathf.Min(joints.Length, Mathf.Min(a.j.Length, b.j.Length) / 7);
            for (int i = 0; i < count; i++)
            {
                if (joints[i] == null)
                    continue;
                Pose local = LerpPose(a.j, b.j, i * 7, alpha);
                joints[i].SetLocalPositionAndRotation(local.position, local.rotation);
            }
        }

        private static Pose LerpPose(float[] a, float[] b, int i, float alpha)
        {
            var pa = new Vector3(a[i], a[i + 1], a[i + 2]);
            var pb = new Vector3(b[i], b[i + 1], b[i + 2]);
            var qa = new Quaternion(a[i + 3], a[i + 4], a[i + 5], a[i + 6]);
            var qb = new Quaternion(b[i + 3], b[i + 4], b[i + 5], b[i + 6]);
            return new Pose(Vector3.Lerp(pa, pb, alpha), Quaternion.Slerp(qa, qb, alpha));
        }

        /// <summary>The transform in `copy` at the same path that `original` has under `sourceRoot`.</summary>
        private static Transform Counterpart(Transform original, Transform sourceRoot, Transform copy)
        {
            if (original == null)
                return null;
            var path = new List<string>();
            for (var t = original; t != sourceRoot; t = t.parent)
            {
                if (t == null)
                    return null;   // not under the HandVisual — can't map it
                path.Insert(0, t.name);
            }
            return path.Count == 0 ? copy : copy.Find(string.Join("/", path));
        }

        /// <summary>Remove every script and collider so the copy is just transforms + renderers.</summary>
        private static void StripScripts(GameObject copy)
        {
            // A component can't be removed while another one requires it. Components that
            // depend on others are usually added later, so remove in reverse order, and
            // repeat in case a dependency ordering still blocked something.
            for (int pass = 0; pass < 5; pass++)
            {
                var components = copy.GetComponentsInChildren<Component>(true);
                int remaining = 0;
                for (int i = components.Length - 1; i >= 0; i--)
                {
                    var component = components[i];
                    if (component is MonoBehaviour || component is Collider)
                    {
                        UnityEngine.Object.DestroyImmediate(component);
                        if (component != null)   // Unity null-check: still alive → blocked
                            remaining++;
                    }
                }
                if (remaining == 0)
                    break;
            }
        }
    }

    private void Shutdown()
    {
        if (hasShutdown) return;
        hasShutdown = true;

        isRunning = false;
        if (receiveThread != null && receiveThread.IsAlive && !receiveThread.Join(1000))
            Debug.LogWarning("[HandReplay] ⚠️ Receive thread did not exit within timeout.");
        if (registered)
            NetMQManager.UnregisterReceiver();
    }

    private void OnDestroy() => Shutdown();
    private void OnApplicationQuit() => Shutdown();
}
