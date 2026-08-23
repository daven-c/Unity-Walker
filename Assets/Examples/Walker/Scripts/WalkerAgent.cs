using System;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using BodyPart = Unity.MLAgentsExamples.BodyPart;
using Random = UnityEngine.Random;

public class WalkerAgent : Agent
{
    [Header("Walk Speed")]
    [Range(0.1f, 5)]
    [SerializeField]
    //The walking speed to try and achieve
    private float m_TargetWalkingSpeed = 5;

    public float MTargetWalkingSpeed // property
    {
        get { return m_TargetWalkingSpeed; }
        set { m_TargetWalkingSpeed = Mathf.Clamp(value, .1f, m_maxWalkingSpeed); }
    }

    //The max walking speed. Was 10 m/s, which is roughly a world-record sprint - the velocity
    //matching reward is the only thing being optimized, so demanding that speed from a ragdoll
    //that can barely walk is satisfied by hurling limbs around. 5 is a jog.
    //Note this also normalizes GetMatchingVelocityReward, so changing it rescales reward magnitudes.
    const float m_maxWalkingSpeed = 5;

    //Should the agent sample a new goal velocity each episode?
    //If true, walkSpeed will be randomly set between zero and m_maxWalkingSpeed in OnEpisodeBegin()
    //If false, the goal velocity will be walkingSpeed
    public bool randomizeWalkSpeedEachEpisode;

    //The direction an agent will walk during training.
    private Vector3 m_WorldDirToWalk = Vector3.right;

    [Header("Target To Walk Towards")] public Transform target; //Target the agent will walk towards during training.

    [Header("Body Parts")] public Transform hips;
    public Transform chest;
    public Transform spine;
    public Transform head;
    public Transform thighL;
    public Transform shinL;
    public Transform footL;
    public Transform thighR;
    public Transform shinR;
    public Transform footR;
    public Transform armL;
    public Transform forearmL;
    public Transform handL;
    public Transform armR;
    public Transform forearmR;
    public Transform handR;

    public bool Inference;

    [Header("Fall Recovery")]
    [Range(0f, 1f)]
    [Tooltip("Chance an episode starts with the ragdoll already knocked over, so it gets practice standing back up.")]
    public float fallenStartProbability = 0.3f;

    [Tooltip("Physics steps the ragdoll may stay collapsed before the episode is cut short. 0 disables. " +
             "Without this a ragdoll that can't recover lies still for the rest of the episode earning ~0. " +
             "Cutting it converts one dead episode into several fresh recovery attempts, since each new " +
             "episode reseeds a different fallen pose. Keep it generous - getting up takes several seconds, " +
             "and too low a value cuts off attempts that were about to succeed.")]
    public int collapsedStepLimit = 600;

    [Range(0f, 1f)]
    [Tooltip("Posture below this counts as collapsed for the timeout above.")]
    public float collapsedPostureThreshold = 0.2f;

    [Range(0f, 110f)]
    [Tooltip("How far a fallen start tips the ragdoll, in degrees of pitch. ~20 is a stumble the " +
             "walking policy can often correct already; 90 is fully prone and much harder. " +
             "Overridden by the 'fallen_tilt' environment parameter, so config.yaml can ramp it as " +
             "a curriculum rather than always training the hardest case.")]
    public float fallenStartTilt = 30f;

    //Consecutive physics steps spent below collapsedPostureThreshold.
    int m_CollapsedSteps;

    [Header("Motion Smoothness")]
    [Range(0f, 0.5f)]
    [Tooltip("Penalty per decision for how much the action vector changed since the last decision. " +
             "Discourages twitchy, oscillating joint commands without penalizing smooth motion. " +
             "Too high and the agent freezes to avoid paying it.")]
    public float actionRatePenalty = 0.05f;

    //Previous decision's continuous actions, for the action-rate penalty above.
    float[] m_PrevActions;

    //This will be used as a stabilized model space reference point for observations
    //Because ragdolls can move erratically during training, using a stabilized reference transform improves learning
    OrientationCubeController m_OrientationCube;

    //The indicator graphic gameobject that points towards the target
    DirectionIndicator m_DirectionIndicator;
    JointDriveController m_JdController;
    EnvironmentParameters m_ResetParams;

    //Head-to-foot height of the prefab's authored (standing) pose, captured in Initialize and used
    //to normalize the posture reward. Self-calibrating, so resizing the ragdoll doesn't need a retune.
    float m_StandingHeight = 1f;

    public override void Initialize()
    {
        m_OrientationCube = GetComponentInChildren<OrientationCubeController>();
        m_DirectionIndicator = GetComponentInChildren<DirectionIndicator>();

        //Setup each body part
        m_JdController = GetComponent<JointDriveController>();
        m_JdController.SetupBodyPart(hips);
        m_JdController.SetupBodyPart(chest);
        m_JdController.SetupBodyPart(spine);
        m_JdController.SetupBodyPart(head);
        m_JdController.SetupBodyPart(thighL);
        m_JdController.SetupBodyPart(shinL);
        m_JdController.SetupBodyPart(footL);
        m_JdController.SetupBodyPart(thighR);
        m_JdController.SetupBodyPart(shinR);
        m_JdController.SetupBodyPart(footR);
        m_JdController.SetupBodyPart(armL);
        m_JdController.SetupBodyPart(forearmL);
        m_JdController.SetupBodyPart(handL);
        m_JdController.SetupBodyPart(armR);
        m_JdController.SetupBodyPart(forearmR);
        m_JdController.SetupBodyPart(handR);

        //Measured before any physics runs, so the ragdoll is still in its authored standing pose.
        var standingHeight = head.position.y - Mathf.Min(footL.position.y, footR.position.y);
        if (standingHeight > 0.01f)
        {
            m_StandingHeight = standingHeight;
        }

        m_ResetParams = Academy.Instance.EnvironmentParameters;
    }

    /// <summary>
    /// Loop over body parts and reset them to initial conditions.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        //Reset all of the body parts
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }

        //Drop the action history so the first decision of the episode isn't charged an
        //action-rate penalty for the discontinuity across the episode boundary.
        m_PrevActions = null;
        m_CollapsedSteps = 0;

        //Random start rotation to help generalize
        var yaw = Random.Range(0.0f, 360.0f);
        if (Random.value < fallenStartProbability)
        {
            //Tip the ragdoll over so it has to practice recovering, rather than only ever starting
            //from a stable standing pose. The tilt is curriculum-driven: recovering from fully prone
            //(~90) is a long precise sequence that random exploration essentially never finds, while
            //a shallow tilt is a stumble the walking policy can already half-correct - which gives
            //it something to climb from. Ramp 'fallen_tilt' in config.yaml as it improves.
            var tilt = m_ResetParams.GetWithDefault("fallen_tilt", fallenStartTilt);
            var pitch = Mathf.Clamp(tilt + Random.Range(-10f, 10f), 0f, 110f);
            hips.rotation = Quaternion.Euler(pitch, yaw, Random.Range(0f, 360f));
        }
        else
        {
            hips.rotation = Quaternion.Euler(0, yaw, 0);
        }

        UpdateOrientationObjects();

        //Set our goal walking speed
        MTargetWalkingSpeed =
            randomizeWalkSpeedEachEpisode ? Random.Range(0.1f, m_maxWalkingSpeed) : MTargetWalkingSpeed;
    }

    /// <summary>
    /// Add relevant information on each body part to observations.
    /// </summary>
    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
    {
        //GROUND CHECK
        sensor.AddObservation(bp.groundContact.touchingGround); // Is this bp touching the ground

        //Get velocities in the context of our orientation cube's space
        //Note: You can get these velocities in world space as well but it may not train as well.
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.linearVelocity));
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.angularVelocity));

        //Get position relative to hips in the context of our orientation cube's space
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(bp.rb.position - hips.position));

        if (bp.rb.transform != hips && bp.rb.transform != handL && bp.rb.transform != handR)
        {
            sensor.AddObservation(bp.rb.transform.localRotation);
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
        }
    }

    /// <summary>
    /// Loop over body parts to add them to observation.
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        var cubeForward = m_OrientationCube.transform.forward;

        //velocity we want to match
        var velGoal = cubeForward * MTargetWalkingSpeed;
        //ragdoll's avg vel
        var avgVel = GetAvgVelocity();

        //current ragdoll velocity. normalized
        sensor.AddObservation(Vector3.Distance(velGoal, avgVel));
        //avg body vel relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(avgVel));
        //vel goal relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(velGoal));

        //rotation deltas
        sensor.AddObservation(Quaternion.FromToRotation(hips.forward, cubeForward));
        sensor.AddObservation(Quaternion.FromToRotation(head.forward, cubeForward));

        //Position of target position relative to cube
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformPoint(target.transform.position));

        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)

    {
        var bpDict = m_JdController.bodyPartsDict;
        var i = -1;

        var continuousActions = actionBuffers.ContinuousActions;
        bpDict[chest].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[spine].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);

        bpDict[thighL].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[thighR].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[shinL].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[shinR].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[footR].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);
        bpDict[footL].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], continuousActions[++i]);

        bpDict[armL].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[armR].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[forearmL].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[forearmR].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[head].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);

        //update joint strength settings
        bpDict[chest].SetJointStrength(continuousActions[++i]);
        bpDict[spine].SetJointStrength(continuousActions[++i]);
        bpDict[head].SetJointStrength(continuousActions[++i]);
        bpDict[thighL].SetJointStrength(continuousActions[++i]);
        bpDict[shinL].SetJointStrength(continuousActions[++i]);
        bpDict[footL].SetJointStrength(continuousActions[++i]);
        bpDict[thighR].SetJointStrength(continuousActions[++i]);
        bpDict[shinR].SetJointStrength(continuousActions[++i]);
        bpDict[footR].SetJointStrength(continuousActions[++i]);
        bpDict[armL].SetJointStrength(continuousActions[++i]);
        bpDict[forearmL].SetJointStrength(continuousActions[++i]);
        bpDict[armR].SetJointStrength(continuousActions[++i]);
        bpDict[forearmR].SetJointStrength(continuousActions[++i]);

        //Penalize how much the action vector moved since the last decision. Nothing else in the
        //reward costs anything for slamming joints to full force or reversing them every frame,
        //so without this the flailing gait scores the same as a smooth one.
        if (m_PrevActions == null || m_PrevActions.Length != continuousActions.Length)
        {
            m_PrevActions = new float[continuousActions.Length];
        }
        else if (actionRatePenalty > 0f)
        {
            var delta = 0f;
            for (var j = 0; j < continuousActions.Length; j++)
            {
                delta += Mathf.Abs(continuousActions[j] - m_PrevActions[j]);
            }

            AddReward(-actionRatePenalty * (delta / continuousActions.Length));
        }

        for (var j = 0; j < continuousActions.Length; j++)
        {
            m_PrevActions[j] = continuousActions[j];
        }
    }

    //Update OrientationCube and DirectionIndicator
    void UpdateOrientationObjects()
    {
        m_WorldDirToWalk = target.position - hips.position;
        m_OrientationCube.UpdateOrientation(hips, target);
        if (m_DirectionIndicator)
        {
            m_DirectionIndicator.MatchOrientation(m_OrientationCube.transform);
        }
    }

    void Update()
    {
        if (Inference && Input.GetMouseButtonDown(0))
        {
            var ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out var hit))
            {
                target.position = new Vector3(hit.point.x, target.position.y, hit.point.z);
            }
        }
    }

    void FixedUpdate()
    {
        UpdateOrientationObjects();

        var cubeForward = m_OrientationCube.transform.forward;

        // Set reward for this step according to mixture of the following elements.
        // a. Match target speed
        //This reward will approach 1 if it matches perfectly and approach zero as it deviates
        var matchSpeedReward = GetMatchingVelocityReward(cubeForward * MTargetWalkingSpeed, GetAvgVelocity());

        //Check for NaNs
        if (float.IsNaN(matchSpeedReward))
        {
            throw new ArgumentException(
                "NaN in moveTowardsTargetReward.\n" +
                $" cubeForward: {cubeForward}\n" +
                $" hips.velocity: {m_JdController.bodyPartsDict[hips].rb.linearVelocity}\n" +
                $" maximumWalkingSpeed: {m_maxWalkingSpeed}"
            );
        }

        // b. Rotation alignment with target direction.
        //This reward will approach 1 if it faces the target direction perfectly and approach zero as it deviates
        var headForward = head.forward;
        headForward.y = 0;
        // var lookAtTargetReward = (Vector3.Dot(cubeForward, head.forward) + 1) * .5F;
        var lookAtTargetReward = (Vector3.Dot(cubeForward, headForward) + 1) * .5F;

        //Check for NaNs
        if (float.IsNaN(lookAtTargetReward))
        {
            throw new ArgumentException(
                "NaN in lookAtTargetReward.\n" +
                $" cubeForward: {cubeForward}\n" +
                $" head.forward: {head.forward}"
            );
        }

        // c. Posture: torso vertical AND actually standing tall on its legs.
        //This GATES the locomotion reward instead of adding to it. Crawling satisfies both
        //matchSpeedReward (average body velocity) and lookAtTargetReward (head yaw) perfectly well,
        //so as a small additive bonus this lost to the worming gait it was meant to discourage.
        //The height term matters too: dot(hips.up, up) alone scores ~1 while lying on your back.
        var torsoUpright = Mathf.Clamp01(Vector3.Dot(hips.up, Vector3.up));
        var height = head.position.y - Mathf.Min(footL.position.y, footR.position.y);
        var postureReward = torsoUpright * Mathf.Clamp01(height / m_StandingHeight);

        //The standalone posture term is what gives a fallen agent a gradient to stand back up,
        //since the gated locomotion term is ~0 until it does.
        AddReward(matchSpeedReward * lookAtTargetReward * postureReward + 0.1f * postureReward);

        //Cut the episode short if the ragdoll has been collapsed for too long. The counter resets
        //the moment posture recovers, so an agent making progress toward standing keeps its time.
        if (collapsedStepLimit > 0)
        {
            m_CollapsedSteps = postureReward < collapsedPostureThreshold ? m_CollapsedSteps + 1 : 0;
            if (m_CollapsedSteps >= collapsedStepLimit)
            {
                m_CollapsedSteps = 0;

                //EpisodeInterrupted, NOT EndEpisode. This resolves to DoneReason.MaxStepReached,
                //which bootstraps the value estimate. EndEpisode would mark a terminal state worth
                //zero future reward, re-teaching "being on the ground is death" - exactly the lesson
                //stage 2 exists to unlearn.
                EpisodeInterrupted();
            }
        }
    }

    //Returns the average velocity of all of the body parts
    //Using the velocity of the hips only has shown to result in more erratic movement from the limbs, so...
    //...using the average helps prevent this erratic movement
    Vector3 GetAvgVelocity()
    {
        Vector3 velSum = Vector3.zero;

        //ALL RBS
        int numOfRb = 0;
        foreach (var item in m_JdController.bodyPartsList)
        {
            numOfRb++;
            velSum += item.rb.linearVelocity;
        }

        var avgVel = velSum / numOfRb;
        return avgVel;
    }

    //normalized value of the difference in avg speed vs goal walking speed.
    public float GetMatchingVelocityReward(Vector3 velocityGoal, Vector3 actualVelocity)
    {
        //distance between our actual velocity and goal velocity
        var velDeltaMagnitude = Mathf.Clamp(Vector3.Distance(actualVelocity, velocityGoal), 0, MTargetWalkingSpeed);

        //return the value on a declining sigmoid shaped curve that decays from 1 to 0
        //This reward will approach 1 if it matches perfectly and approach zero as it deviates
        return Mathf.Pow(1 - Mathf.Pow(velDeltaMagnitude / MTargetWalkingSpeed, 2), 2);
    }

    /// <summary>
    /// Agent touched the target
    /// </summary>
    public void TouchedTarget()
    {
        AddReward(1f);
    }
}
