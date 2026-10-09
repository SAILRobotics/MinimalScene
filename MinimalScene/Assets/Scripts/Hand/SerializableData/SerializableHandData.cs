using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;

using SerializableData;


namespace SerializableData 
{

    [Serializable]
    public class SerializableHandData 
    {
        public Dictionary<string, List<SerializablePose>> groups = new();
        public float indexPinchStrength = 0f;

        // What HandVisual needs to pose the hand mesh: the root pose (world), the hand
        // scale, and every joint's pose relative to its parent bone, in HandJointId order.
        public SerializablePose rootPose;
        public float scale = 1f;
        public List<SerializablePose> localJoints;
    }

}