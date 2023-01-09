Appendix - Tissue properties
---
The complex composition of the skull bone causes significant distortion of the ultrasound transmission into the brain cavity. For purposes of To this date, there is not yet a definitive method to produce maps of acoustic properties (density, speed of sound and attenuation).
# Basic material properties

| Material type | Long. Speed of sound (m/s) | Density (kg/m$^3$)| Long. attenuation (Np/m)$^*$|
|-------|---------|------|-------|
| Water | 1500 | 1000 | 0 |
| Skin | 1500 | 1000 | 4.6 $f^{1.0}$ |
| Brain| 1610 | 1090 | 6.9 $f^{1.0}$  |

$^*f$ = frequency in MHz
Reference: Aubry *et al.* [DOI:10.1121/10.0013426](https://doi.org/10.1121/10.0013426)
## Simple skull mask - (no CT/ZTE scan)
| Material type | Long. Speed of sound (m/s) | Density (kg/m$^3$)| Long. attenuation (Np/m)*|
|-------|---------|------|-------|
| Cortical| 120$f$ + 2416 $^*$  |     1896.5 |   |
| Trabecular| 282$f$ + 2063 $^*$  |1738.0 | |

Reference: $^*$Pichardo *et al.* [10.1088/0031-9155/56/1/014](https://doi.org/10.1088/0031-9155/56/1/014)

## CT based mapping
Because of its relationship with bone density, computed tomography (CT) Hounsfield units (HU) scans of skull bone have been used as a surrogate to calculate acoustic properties maps suitable for transcranial ultrasound modeling. For a very comprehensive review of modeling efforts including different approaches to mapping HU to acoustic properties, please consult Angla *et al.* (2002) "*Transcranial ultrasound simulations: A review*" [10.1002/mp.15955](https://doi.org/10.1002/mp.15955).
 
While the use of CT maps has been successful to facilitate the development of procedures that became approved for clinic use (Elias *et al.* 2016, [DOI:10.1056/NEJMoa1600159](https://doi.org/10.1056/NEJMoa1600159)), it has been in a very specific set of conditions. Webb *et al.* presented two studies detailing clearly how XRays energy level, scan model and CT kernel reconstruction influence acoustic properties derived from HU ([DOI:10.1109/TUFFC.2018.2827899](https://doi.org/10.1109/TUFFC.2018.2827899), [DOI:10.1109/TUFFC.2020.3039743](https://doi.org/10.1109/TUFFC.2020.3039743)).

BabelBrain uses a mapping of properties proposed by 

The choice of this specific mapping was done after performing a test with five CT scans representing a sampling of skull density ratio (SDR). The SDR is a metric that measures the HUs of trabecular bone over cortical. The SDR is a very important metric used in the treatment of essential tremor with MRI-guided focused ultrasound