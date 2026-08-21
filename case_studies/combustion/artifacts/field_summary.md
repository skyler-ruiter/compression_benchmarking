# S3D sampled field summary

Source: `/media/volume/Compression_Data/sdrbench_data/S3D_500x500x500/vars_500x500x500`  
Samples per field: 500,000  
Values below are converted to physical units where metadata supplies a scale.

| field | unit | min | median | max | mean | std | XY gradient p99 / range |
|---|---:|---:|---:|---:|---:|---:|---:|
| CH4 | mass fraction | -6.3615e-15 | -4.4847e-15 | 0.0392 | 0.015463 | 0.018413 | 0.037359 |
| O2 | mass fraction | 0.067534 | 0.067565 | 0.2239 | 0.12923 | 0.073436 | 0.037251 |
| CO | mass fraction | -1.0355e-15 | 8.4542e-05 | 0.00045396 | 6.0399e-05 | 6.6784e-05 | 0.092111 |
| CO2 | mass fraction | -1.0355e-15 | 0.1074 | 0.10743 | 0.065031 | 0.050446 | 0.036981 |
| H2O | mass fraction | -1.0355e-15 | 0.088045 | 0.088063 | 0.053319 | 0.04136 | 0.03735 |
| N2 | mass fraction | 0.73689 | 0.7369 | 0.7369 | 0.7369 | 3.7011e-06 | 0.0073862 |
| TEMP | K | 299.99 | 1837.9 | 1843.4 | 1237.1 | 720.47 | 0.036078 |
| PRES | Pa | 1.0129e+05 | 1.0132e+05 | 1.0133e+05 | 1.0132e+05 | 1.7191 | 0.0072432 |
| U | m/s | -3.4278 | 1.3447 | 6.8083 | 0.9691 | 1.0095 | 0.028091 |
| V | m/s | -3.979 | -0.083568 | 3.7281 | -0.091042 | 0.64789 | 0.024816 |
| W | m/s | -2.9589 | -0.0024447 | 4.2723 | -0.082635 | 0.63251 | 0.035243 |

The gradient proxy selects the top 10% of temperature-gradient pixels on the central XY plane. It is diagnostic only, not a validated flame mask.
