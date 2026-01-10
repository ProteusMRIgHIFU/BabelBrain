
---
title: "Information for driving equipment with BabelBrain output files"
author: "Samuel Pichardo, Associated Professor,University of Calgary"
date: "January 9, 2026"
numbersections: true
---
# Introduction
This document presents information aimed for the development of interfaces to drive ultrasound equipment using BabelBrain output data.

Files generated at the end of execution of Thermal Modeling (Step 3) can be used to program driving equipment. In specific, the HDF5 file ending with the `...DataForSim-ThermalField_AllCombinations.h5` suffix, which concentrates all the necessary information. Below it is detailed the useful entries in this HDF5 file, such as `/RatioLosses`, that can be used to program the driving equipment.

There are four majors components that can be used to program the driving equipment:

1. Intensity calibration 
2. Steering conditions (if applicable)
3. Timing parameters
4. Phase data (if applicable)

# Intensity calibration
The  `/RatioLosses` entry is a scalar value between 0.0 and 1.0 describing the energy loss  after considering all mechanisms (reflection and attenuation) that is used in BabelBrain to calculate the $I_{\text{SPPA Water}}$ needed to achieve a desired $I_{\text{SPPA Brain}}$. For example, let's assume the user is aiming to achieve $I_{\text{SPPA Brain}} = 5 \text{ W/cm}^2$ and that the `/RatioLosses` entry has a value of 0.105. In that case,

$$
I_{\text{SPPA Water}} = \frac{I_{\text{SPPA Brain}}}{\text{RatioLosses}}=\frac{5.0 \text{ W/cm}^2}{0.105}=47.6 \text{ W/cm}^2
$$

The driving equipment should be programmed to achieve 47.6 $\text{W/cm}^2$ in water at the focus, which is the most common method transducers are calibrated.

# Steering conditions
Two types of steering conditions could occur depending on the transducer type: Concentric focusing annular array or phased array. For single element transducers, there are no applicable steering conditions.

##  Concentric annular array
These devices can be a spherical cap (i.e. CTX-500) or flat device (i.e. H246) cut in annular segments, allowing focusing over the Z direction. For this case, the HDF5 entry `/ZSteering` (in meters) will have the steering to apply depending on the subtype of device.

### Spherical cap
The `/ZSteering` entry is relative to the natural focus. If `/ZSteering` is positive, that means the steering is deeper, farther away from the transducer. If `/ZSteering` is negative, that means the steering is shallower, closer to the transducer. 

### Flat device
The `/ZSteering` entry is relative to the surface of the device, and it is always positive following the same convention as for the spherical cap.

## Phased arrays
Phased arrays can steer in 3 directions and follow an analogous convention as for the concentric annular arrays. They can also a spherical cap (i.e. H317) or flat device (i.e. REMOPD)

### Spherical cap
* `/ZSteering` is relative to the natural focus. If `/ZSteering` is positive, that means the steering is deeper, farther away from the transducer. If `/ZSteering` is negative, that means the steering is shallower, closer to the transducer. 
* `/XSteering` (in meters) is relative to the acoustic axis and the direction convention is dictated by the transducer definition. The positive direction is towards the direction where the transducer elements have a positive value, and vice versa.
* `/YSteering` (in meters) is relative to the acoustic axis and the direction convention is dictated by the transducer definition. The positive direction is towards the direction where the transducer elements have a positive value, and vice versa.

This convention means the X and Y directions are primarily relative to the transducer elements position **and not to the patient coordinate system**. It is the responsibility of the user to ensure to verify the orientation of the device between simulations and real experiments to ensure steering is applied in the desired direction in the patient coordinates space.

###  Flat device
* `/ZSteering` is relative to the surface of the device, and it is always positive following the same convention as for the spherical cap.
* `/XSteering` and `/YSteering` entries are identical as for spherical caps.

## Timing parameters
The timing parameters are initially specified by the user via a YAML file as the one shown below:

```YAML
AllDC_PRF_Duration: #All combinations of timing that will be considered
    -   DC: 0.3
        PRF: 10.0
        Duration: 40.0
        DurationOff: 40.0
```
This example is for a sonication with a duration of 40s, with PRF = 10 Hz, DC = 0.3 plus an off period (ultrasound completely turned off) of 40s. These parameters can be used to program the timing of the exposure with the driving equipment.

This definition can be extended to allow repetitions. For example:

```YAML
AllDC_PRF_Duration: #All combinations of timing that will be considered
    -   DC: 0.3
        PRF: 10.0
        Duration: 40.0
        DurationOff: 40.0
        Repetitions: 10
```

In this example, the previous protocol will be repeated 10 times (including the off time).

Finally, this can be even further extended to grouped sonications with extra dead time. For example:

```YAML
AllDC_PRF_Duration: #All combinations of timing that will be considered
    -   DC: 0.3
        PRF: 10.0
        Duration: 40.0
        DurationOff: 40.0
        Repetitions: 10
        NumberGroupedSonications: 4
        PauseBetweenGroupedSonications: 360.0
```

In this example, the ensemble of the 10 repetitions will be further repeated 4 more times, adding an extra off time of 360 seconds between groups. 

### Multiple protocols
The user can specify in the same YAML file multiple timing protocols. For example:
```YAML
AllDC_PRF_Duration: #All combinations of timing that will be considered
    -   DC: 0.3
        PRF: 10.0
        Duration: 40.0
        DurationOff: 40.0
    -   DC: 0.1
        PRF: 5.0
        Duration: 80.0
        DurationOff: 50.0
```
In this case, two different protocols are simulated. This is useful for users desiring to study different approaches.

### Timing information in HDF5 file
All the previous information is stored in the HDF5 file. The `/AllData` subdirectory entry in the file contains each protocol defined in the YAML file as subdirectories such as:

* `/AllData/item_0`
* `/AllData/item_1`
* etc.

Each subdirectory will contain the timing information, such as `/AllData/item_0/DurationUS`, of each protocol with the entries:

* `/DurationUS`: duration in seconds of the ultrasound exposure
* `/PRF`: Pulse repetition frequency in Hz
* `/DutyCycle`: Fraction between 0.0 an 1.0.
* `/DurationOff`: Duration in seconds of ultrasound turned off completely.
* `/Repetitions`: Number of repetitions as described above. It will have a vale of 1 if not specified by the user in the YAML file.
* `/NumberGroupedSonications`: Number of repetitions of the ensemble as described above. It will have a vale of 1 if not specified by the user in the YAML file.
* `/PauseBetweenGroupedSonications`: Extra off time in seconds between grouped sonications as described above. It will have a vale of 0 if not specified by the user in the YAML file.

# Phase data
Files for ring-type arrays and phased arrays will store phase programming information as follows:

* `/PhaseData`: This is an array of N-elements for ring-type arrays and phased arrays with the complex values used in the simulation to program the device without any skull refocusing compensation. This is the most common operation as skull refocusing depends on users having tight control of the orientation of the transducer between planning and real experiments. A driving equipment may use `angle(PhaseData)` to program the device. However, often software controlling the driving equipment uses `ZSteering` (and `XSteering` and `YSteering` if applicable) to program the device. If `PhaseData` is used to program the device, the steering variables should be ignored, and vice versa.

* `/PhaseDataRefocusing`: This is an optional array of N-elements only available for phased arrays with the complex values used in the simulation to program the device to compensate the skull effects. The user must select "refocusing" in Step 2 to make this data available. Use this only if the user has tight control of the position of the transducer between simulation and real experiments. A driving equipment may use `angle(PhaseDataRefocusing)` to program the device.  If `PhaseDataRefocusing` is used to program the device, the steering variables should be ignored.