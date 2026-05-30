**Label APS**
| Label             | 0.5      | 1.0      | 2.0      | 4.0      |
|-------------------|----------|----------|----------|----------|
| car               |  0.270  |  0.598  |  0.818  |  0.891  |
| truck             |  0.000  |  0.000  |  0.000  |  0.000  |
| bus               |  0.000  |  0.000  |  0.000  |  0.000  |
| trailer           |  0.000  |  0.000  |  0.000  |  0.000  |
| construction_vehicle |  0.000  |  0.000  |  0.000  |  0.000  |
| pedestrian        |  0.103  |  0.208  |  0.297  |  0.336  |
| motorcycle        |  0.206  |  0.473  |  0.607  |  0.657  |
| bicycle           |  0.000  |  0.000  |  0.000  |  0.000  |
| traffic_cone      |  0.000  |  0.000  |  0.000  |  0.000  |
| barrier           |  0.000  |  0.000  |  0.000  |  0.000  |

**Mean Dist APS**
| Label             | Mean Dist AP |
|-------------------|--------------|
| car               |  0.644        |
| truck             |  0.000        |
| bus               |  0.000        |
| trailer           |  0.000        |
| construction_vehicle |  0.000        |
| pedestrian        |  0.236        |
| motorcycle        |  0.486        |
| bicycle           |  0.000        |
| traffic_cone      |  0.000        |
| barrier           |  0.000        |

**mAP**
| mAP |
|-----|
|  0.137 |

**TP Errors (Translational, Scale, Orientation, Velocity, Attribute)**
| Label             | Trans Err | Scale Err | Orient Err | Vel Err | Attr Err |
|-------------------|-----------|-----------|------------|---------|----------|
| car               |  0.444     |  0.142     |  0.040      |  5.078   |  1.000    |
| truck             |  1.000     |  1.000     |  1.000      |  1.000   |  1.000    |
| bus               |  1.000     |  1.000     |  1.000      |  1.000   |  1.000    |
| trailer           |  1.000     |  1.000     |  1.000      |  1.000   |  1.000    |
| construction_vehicle |  1.000     |  1.000     |  1.000      |  1.000   |  1.000    |
| pedestrian        |  0.660     |  0.315     |  0.797      |  1.102   |  1.000    |
| motorcycle        |  0.585     |  0.262     |  0.276      |  1.541   |  1.000    |
| bicycle           |  1.000     |  1.000     |  1.000      |  1.000   |  1.000    |
| traffic_cone      |  1.000     |  1.000     |  nan      |  nan   |  nan    |
| barrier           |  1.000     |  1.000     |  1.000      |  nan   |  nan    |

**Average TP Errors Across All Labels (excluding NaNs)**
| Trans Err | Scale Err | Orient Err | Vel Err | Attr Err |
|-----------|-----------|------------|---------|----------|
|  0.869     |  0.772     |  0.790      |  1.590   |  1.000    |

**TP Scores (Translational, Scale, Orientation, Velocity, Attribute)**
| Metric            | TP Score   |
|-------------------|------------|
| trans_err         |  0.131      |
| scale_err         |  0.228      |
| orient_err        |  0.210      |
| vel_err           |  0.000      |
| attr_err          |  0.000      |

**ND Score**
| ND Score  |
|-----------|
|  0.125     |

**Evaluation Time**
| Evaluation Time (s) |
|---------------------|
|  51.790              |