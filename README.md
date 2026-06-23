A system is built which can be used in vehicles to make sure that the driver doesnt get into sleep or fatigue-induced accidents. The current system is a dual phase system split into two parts based on the scenario of the vehicle.
Phase 1: Pre-Ignition
    In this phase, the key has been inserted and the car has been started giving power to the camera. This model takes the footage from the camera and cross verifies the person in the driver seat against a pre-authorized database of drivers which will lead to 2 scenarios - 
    Scenario 1 - Authorised Person
        If the person in the frame has been recognised as an authorised person in the database then it will go to Phase 2 - Real-Time Liveness Detection and allows ignition of the vehicle.
    Scenario 2 - Unauthorised Person
        If the person in the frame is not in the pre-authorised database, then a notification will be sent to the owner of the vehicle for verification. if the owner allows permission of the person, then the face will be added to the database and allows for ignition of the vehicle and moves to Phase 2

Phase 2: Transit Phase
    This is the main feature of the entire project. In this phase, during transit, the camera placed will take real-time frames from the footage, these frames will be used in 3 aspects - calculating EAR (Eye Aspect Ratio), MAR (Mouth Aperture Ration) and Head Drops looking for a sudden deceleration in face landmarks on a 2d scale by comparing it with a normalized human face.
    Aspect 1 - EAR
        EAR or Eye Aspect Ratio is used to detect if a person's eyes are awake, or sleepy. Landmarks around the left and right eye will be taken into consideration and distance between the landmarks is taken and the EAR values of left and right eye is taken and averaged. If the averaged EAR value falls below a certain threshold, then the event is triggered wherein a buzzer sound will be given out to wake the person.
        To track micro-sleeps and involuntary eyelid closures, the system isolates six specific localized coordinate point indices around the eye perimeters. The EAR maps the vertical distance between the eyelids relative to the horizontal width of the eye:

        EAR=  (| p2-p6 |+| p3-p5 |)/(2×| p1-p4|)

        Where p1 through p6 represent the spatial coordinate positions outputted by the mesh layer. Under normal alert conditions, the EAR value hovers at a stable baseline unique to the driver. When a micro-sleep occurs, the vertical landmarks collapse toward each other, driving the EAR toward zero. The system flags a definitive drowsiness state if the calculated EAR breaks beneath a calibrated threshold (approximately 0.20) and remains flatlined across a consecutive rolling frame window exceeding 1.5 seconds (equivalent to roughly 45 consecutive frames).

    Aspect 2 - MAR
        MAR or Mouth Aperture Ratio is used to detect if a person is yawning which is an indicator of fatigue or drowsyness. Landmarks around the persons mouth is taken and if this value exceeds a certain threshold, then the event is triggered and the same buzzer sound will be given out to wake the person. This, however, has a precondition of EAR and only if both conditions are true, it will give the buzzer and the person opening their mouth to talk would also be triggered if that was the case. When a person usually yawns, their eyes close a bit too but that if isnt the case when the person is talking.
        Drivers experiencing severe exhaustion often exhibit prolonged periods of staring blankly ahead with zero natural micro-blinks, a state known as zoning out, or frequent, deep yawning. To capture these fatigue indicators, the system monitors the Mouth Aperture Ratio (MAR):

        MAR=  (| m2- m6 |+| m3-m5 |)/(2 ×| m1-m4 |)

        Where m1 through m6 track internal and external lip points. A sustained, hyper-extended spike in the MAR indicates a deep yawn. Conversely, if the variance of both the EAR and MAR values approaches zero over an extended time frame, it indicates a frozen, blank highway stare, allowing the system to flag cognitive detachment even when the driver's eyes remain technically open.

    Aspect 3 - Head Drops
        When a driver slips from deep drowsiness into a total micro-sleep, the neck muscles relax, causing the head to drop forward or tilt heavily to the side. To detect this physical collapse, the system implements Head Pose Estimation by solving the classical Perspective-n-Point (solvePnP) problem.
        The system maps a subset of 3D facial landmarks (including the nose tip, chin, inner eye corners, and mouth corners) against a normalized, rigid 3D model of a human face. By computing the translation and rotation vectors between these coordinate frames, the system derives the exact rotational angles of the driver's head along three primary geometric axes: Pitch (θ), Yaw (ψ), and Roll (φ). A sharp, sudden downward dive along the Pitch axis (θ < –7o) confirms a structural head-drop anomaly, immediately triggering an alert state.
 