using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;

using Oculus.Interaction;
using Oculus.Interaction.Input;


public static class TrackingUtils
{
    public static readonly Dictionary<string, HandJointId[]> HAND_JOINTS = new Dictionary<string, HandJointId[]> {
        { "Wrist", new[] {HandJointId.HandWristRoot} },  
        { "Palm", new[] {HandJointId.HandPalm} }, 
        { "Thumb", new[] {HandJointId.HandThumb1, HandJointId.HandThumb2, HandJointId.HandThumb3, HandJointId.HandThumbTip} }, 
        { "Index", new[] {HandJointId.HandIndex1, HandJointId.HandIndex2, HandJointId.HandIndex3, HandJointId.HandIndexTip} }, 
        { "Middle", new[] {HandJointId.HandMiddle1, HandJointId.HandMiddle2, HandJointId.HandMiddle3, HandJointId.HandMiddleTip} }, 
        { "Ring", new[] {HandJointId.HandRing1, HandJointId.HandRing2, HandJointId.HandRing3, HandJointId.HandRingTip} }, 
        { "Pinky", new[] {HandJointId.HandPinky1, HandJointId.HandPinky2, HandJointId.HandPinky3, HandJointId.HandPinkyTip} }
    };

    // Joints flattened in HAND_JOINTS order: [0] Wrist, [1] Palm, [2-5] Thumb, [6-9] Index,
    // [10-13] Middle, [14-17] Ring, [18-21] Pinky. Matches _HAND_BONES in main_hands_simple.py.
    public const int JOINT_COUNT = 22;
    public static readonly int[,] HAND_BONES = {
        {0, 1}, {0, 2}, {2, 3}, {3, 4}, {4, 5},
        {0, 6}, {6, 7}, {7, 8}, {8, 9},
        {0, 10}, {10, 11}, {11, 12}, {12, 13},
        {0, 14}, {14, 15}, {15, 16}, {16, 17},
        {0, 18}, {18, 19}, {19, 20}, {20, 21},
    };
}
