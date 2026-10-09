using UnityEngine;
using UnityEngine.Rendering;
using NetMQ;
using NetMQ.Sockets;
using System;
using System.Threading;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Newtonsoft.Json;

/// <summary>
/// Receives hand recordings from main_hands_simple.py (L / R keys) and plays each one
/// back once as a ghost skeleton, then hides it. Left and right play independently.
/// Also tints the live hand markers (TrackingDataVisualizer) while Python is recording
/// that hand.
///
/// Recordings are stored relative to WorldRoot (ArUco marker 100), so put this on
/// WorldRoot — or set replayRoot to it — and the ghost replays on the same physical spot.
///
/// Messages (JSON, SUB bound on tcp://0.0.0.0:port):
///   {"type": "replay", "hand": "left"|"right", "frames": [{"t": s, "p": [22 * xyz]}]}
///   {"type": "record_state", "hand": "left"|"right", "recording": bool, "color": [r, g, b]}
/// </summary>
public class HandReplayReceiver : MonoBehaviour
{
    [Header("Targets")]
    [Tooltip("Parent for the ghost skeletons — WorldRoot. Defaults to this transform.")]
    [SerializeField] private Transform replayRoot;
    [Tooltip("Live hand markers to tint while recording. Found automatically if empty.")]
    [SerializeField] private TrackingDataVisualizer liveVisualizer;

    [Header("Ghost skeleton")]
    [Tooltip("Joint marker prefab, e.g. Assets/Scripts/Hand/FingerJoint.")]
    [SerializeField] private GameObject jointPrefab;
    [SerializeField] private Color leftColor  = new Color(0.3f, 0.6f, 1.0f, 1f);
    [SerializeField] private Color rightColor = new Color(1.0f, 0.55f, 0.1f, 1f);
    [SerializeField] private float boneWidth = 0.004f;

    [Header("NetMQ")]
    [SerializeField] private int port = 6003;

    [Header("Debug")]
    [SerializeField] private bool log = true;

    private class ReplayFrame
    {
        public float t;
        public float[] p;
    }

    private class Message
    {
        public string type;
        public string hand;
        public bool recording;
        public float[] color;
        public List<ReplayFrame> frames;
    }

    private static readonly int BaseColorID = Shader.PropertyToID("_BaseColor");
    private static readonly int ColorID = Shader.PropertyToID("_Color");

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
        if (jointPrefab == null)
            Debug.LogWarning("[HandReplay] jointPrefab is not assigned — using plain spheres.", this);

        leftGhost  = new Ghost("ReplayLeftHand",  replayRoot, jointPrefab, leftColor,  boneWidth);
        rightGhost = new Ghost("ReplayRightHand", replayRoot, jointPrefab, rightColor, boneWidth);

        NetMQManager.RegisterReceiver();
        registered = true;
        isRunning = true;
        receiveThread = new Thread(ReceiveLoop) { IsBackground = true };
        receiveThread.Start();

        if (log) Debug.Log($"[HandReplay] 📡 Listening on tcp://0.0.0.0:{port}");
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
                if (ghost.Play(msg.frames) && log)
                    Debug.Log($"[HandReplay] ▶️ {msg.hand} hand: {msg.frames.Count} frames, " +
                              $"{msg.frames[msg.frames.Count - 1].t:F1} s");
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

    /// <summary>One hand's ghost skeleton: joint markers + bone lines, played once then hidden.</summary>
    private class Ghost
    {
        private readonly GameObject root;
        private readonly Transform[] joints = new Transform[TrackingUtils.JOINT_COUNT];
        private readonly LineRenderer[] bones;
        private List<ReplayFrame> frames;
        private float startTime;
        private int cursor;

        public Ghost(string name, Transform parent, GameObject jointPrefab, Color color, float boneWidth)
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

        public bool Play(List<ReplayFrame> newFrames)
        {
            if (newFrames == null || newFrames.Count == 0)
                return false;
            frames = newFrames;
            startTime = Time.time;
            cursor = 0;
            Apply(frames[0].p, frames[0].p, 0f);
            root.SetActive(true);
            return true;
        }

        public void Tick()
        {
            if (frames == null)
                return;

            float t = Time.time - startTime;
            if (t > frames[frames.Count - 1].t)
            {
                root.SetActive(false);   // played once — disappear
                frames = null;
                return;
            }

            while (cursor + 1 < frames.Count && frames[cursor + 1].t <= t)
                cursor++;
            var a = frames[cursor];
            var b = frames[Mathf.Min(cursor + 1, frames.Count - 1)];
            float span = b.t - a.t;
            Apply(a.p, b.p, span > 1e-5f ? (t - a.t) / span : 0f);
        }

        private void Apply(float[] from, float[] to, float alpha)
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
