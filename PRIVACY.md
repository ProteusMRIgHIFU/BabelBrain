# BabelBrain Privacy Policy

**Effective date:** 2026-06-10.  
**Applies to:** the BabelBrain desktop application (macOS, Windows, Linux) and its installers version 0.8.2 and above.  
**Maintained by:** NeuroFUS laboratory, University of Calgary.  
**Contact:** samuel.pichardo@ucalgary.ca  

BabelBrain is an open-source desktop application for prospective modeling of
transcranial focused ultrasound. This policy explains what information the
application does and does not collect.

## Summary

- BabelBrain runs **locally on your computer**. The medical images, trajectories,
  simulation inputs and results you work with are processed on your machine and
  are **never uploaded to us** or to any third party.
- The application collects **no usage information by default**.
- An **optional, opt-in** telemetry feature lets you voluntarily share a small
  amount of **anonymous** technical information to help the project demonstrate
  real-world use to its academic funding agencies. It is **disabled unless you
  explicitly turn it on**, and you can change or turn it off at any time.

## Information processed locally (never collected by us)

BabelBrain reads and writes medical imaging data (e.g. CT/MRI NIfTI files),
trajectory/transform files, configuration files and simulation outputs. All of
this stays on your device (or on storage you control). We do **not** receive,
transmit, or store:

- patient, subject, or any personal/health information;
- the contents of any file you open, generate, or save;
- file names, file paths, or directory contents;
- your name, email address, or account information;
- your IP address (see "What we do not collect", below).

## Optional telemetry (opt-in)

To help us report real-world usage to academic funders and to understand the
tool's performance across hardware, BabelBrain can optionally send a small amount
of anonymous technical data. This feature is **off by default**. The first time
you run the application you are asked, once, whether you wish to participate, and
you choose one of the following levels. You can review or change your choice at
any time under **Advanced Options → Telemetry**.

| Level | What is shared |
|-------|----------------|
| **L0 — No telemetry (default)** | Nothing is sent. |
| **L1 — Basic** | A signal that the app ran, plus CPU model, operating system, amount of main memory, GPU model, and error reports. |
| **L2** | Everything in L1, plus the execution times of the three main simulation steps. |
| **L3** | Everything in L2, plus the selected ultrasound frequency, points-per-wavelength (PPW), simulation domain size, and granular timings of the most computationally demanding sections. |
| **L4** | Everything in L3, plus the transducer model selected and the total run duration (no further timing detail). |

Each telemetry submission is tagged with a **random install identifier** (a
UUID generated on your machine and stored locally). This identifier lets us avoid
double-counting the same installation; it is **not derived from, and cannot be
tied back to, you or your device's identity**.

### What we do not collect, even with telemetry enabled

- No personal, patient, or health information.
- No file contents, file names, or paths.
- No images, trajectories, or simulation results.
- No IP address. BabelBrain does not collect, transmit, or store your IP address
  as part of telemetry. (As with any internet request, the receiving service
  must momentarily handle the network connection to deliver the data; that
  connection metadata is not part of, or retained in, the telemetry dataset.)

## Example of collected data

The table below shows an example of the data submitted with telemetry level L4
for one session. Each row is one event: `install_id` is the random install identifier, `session_date` the start of
the session, `time` the seconds elapsed since the application started, and
`event` the reported information.

<div class="bb-dataframe" markdown>

| install_id | session_date | time | event | app | version |
|---|---|---:|---|---|---|
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 0.000002 | CTS:L1: BabelBrain started | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 0.011065 | CTS:L1: OS: macOS-26.7.1-arm64-arm-64bit-Mach-O | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 0.027903 | CTS:L1: CPU: Apple M3 Max | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 0.027904 | CTS:L1: Memory: 64.00 GB | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 0.027905 | CTS:L1: GPU: Metal Apple M3 Max | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 0.027907 | CTS:L4: Device: DPX_500 | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 3.439590 | CTS:L3:S1: Frequency=500000.0 PPW=6.0 | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 19.415401 | Code block 'CTS:L3:S1: Checking if previously generated files can be reused' took: 0.33770 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 23.411093 | Code block 'CTS:L3:S1: voxelization ' took: 2.19301 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 27.569697 | Code block 'CTS:L3:S1: skull voxelization' took: 0.92616 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 28.203512 | Code block 'CTS:L3:S1: brain voxelization' took: 0.69537 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 29.677689 | Code block 'CTS:L3:S1: skin voxelization' took: 1.47602 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 34.962826 | Code block 'CTS:L3:S1: skin masking' took: 4.91437 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 37.800341 | Code block 'CTS:L3:S1: skull masking' took: 2.79470 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 39.993431 | Code block 'CTS:L3:S1: csf masking' took: 2.01907 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 42.832181 | Code block 'CTS:L3:S1: median filter CT/Density' took: 1.74631 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 47.034216 | Code block 'CTS:L3:S1: binary closing CT/Density' took: 4.16272 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 48.607544 | Code block 'CTS:L3:S1: label CT' took: 1.55786 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 48.713048 | Code block 'CTS:L3:S1: CT extrapol' took: 8.67159 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 49.867536 | Code block 'CTS:L3:S1: CT/Density binary_dilation' took: 1.13351 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 51.128718 | Code block 'CTS:L3:S1: Labeling' took: 1.10292 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 51.549555 | Code block 'CTS:L3:S1: regionprops' took: 0.42537 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 53.017380 | Code block 'CTS:L3:S1: Mapping unique values' took: 0.43121 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 56.772231 | Code block 'CTS:L3:S1: Extracting air regions' took: 3.80799 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 58.506497 | Code block 'CTS:L3:S1: final median filter ' took: 1.73021 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 59.718390 | Code block 'CTS:L3:S1: Second Labeling' took: 1.08972 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 60.124408 | Code block 'CTS:L3:S1: second regionprops' took: 0.47749 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 62.523632 | Code block 'CTS:L3:S1: Final cleanup of prefocal region' took: 1.40698 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 64.996263 | Code block 'CTS:L3:S1: resampling T1 to mask' took: 2.46365 s | BabelBrain | 0.8.8 |
| f4b67d9a-0aa3-4d5c-9066-721e18f15283 | 20261023_195740 | 71.681926 | CTS:L2:S1: TOTAL TIME 67.29732203483582 | BabelBrain | 0.8.8 |

</div>

## How telemetry is transmitted and stored

When telemetry is enabled, the selected data is sent over an encrypted (HTTPS)
connection to a data-collection form hosted by **Google (Google Forms)**, which
acts as our data processor for storing the responses. We use the collected,
aggregated data solely to:

- demonstrate adoption and real-world usage to our funding agencies;
- understand and improve the application's performance and reliability.

We do not sell telemetry data, use it for advertising, or attempt to
re-identify contributors. Google's handling of the underlying form
infrastructure is governed by Google's Privacy Policy
(https://policies.google.com/privacy).

## Legal basis and your control

- Participation is **voluntary and consent-based** (opt-in). No telemetry is sent
  unless you select a level above L0.
- You can **change or withdraw** your choice at any time in
  **Advanced Options → Telemetry** by selecting "L0 — No telemetry".
- Because the data is anonymous and tied only to a random install identifier, we
  generally cannot locate or delete an individual past submission. If you have a
  specific concern, contact us at the address above and we will assist where we
  are able.

## Crash and error reports

Error information included in telemetry (L1 and above) consists of technical
diagnostic messages about failures in the application. We make reasonable efforts
to avoid including any personal or file-specific information in these reports.

## Changes to this policy

We may update this policy as the application evolves (for example, if the
telemetry mechanism or the data categories change). Material changes will be
reflected in this document, with an updated "Effective date." The current version
is always available in the project's source repository.

## Contact

Questions about this policy or about BabelBrain's data practices can be sent to
samuel.pichardo@ucalgary.ca. The project source code is available at
https://github.com/ProteusMRIgHIFU/BabelBrain.
