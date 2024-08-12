# AttrackZone
These are the scripts utilized by *AttrackZone* in order to compile usable projector zones and conduct tracker hijacking attacks. They are intended to provide greater understanding on how *AttrackZone* works.

Video examples of attacks be viewed here:
https://www.youtube.com/playlist?list=PL1wf-CLdUk8KFhgFAHHfaUaku-8IL3z_h

RUNNING ATTACK:

The main file for running the attack is test_hijack_attack.py.  One can pass in datasets of video + LiDAR, or provide video + a segementation model to estimate the attack zones.  It also takes in a Siamese tracker (net), and an optional additional Siamese tracker (net2) to test transferability.
