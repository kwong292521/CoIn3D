**Label APS**
| Label             | 0.5      | 1.0      | 2.0      | 4.0      |
|-------------------|----------|----------|----------|----------|
| car               |  0.195  |  0.496  |  0.751  |  0.852  |
| truck             |  0.000  |  0.000  |  0.000  |  0.000  |
| bus               |  0.000  |  0.000  |  0.000  |  0.000  |
| trailer           |  0.000  |  0.000  |  0.000  |  0.000  |
| construction_vehicle |  0.000  |  0.000  |  0.000  |  0.000  |
| pedestrian        |  0.170  |  0.358  |  0.502  |  0.588  |
| motorcycle        |  0.072  |  0.205  |  0.292  |  0.337  |
| bicycle           |  0.000  |  0.000  |  0.000  |  0.000  |
| traffic_cone      |  0.000  |  0.000  |  0.000  |  0.000  |
| barrier           |  0.000  |  0.000  |  0.000  |  0.000  |

**Mean Dist APS**
| Label             | Mean Dist AP |
|-------------------|--------------|
| car               |  0.573        |
| truck             |  0.000        |
| bus               |  0.000        |
| trailer           |  0.000        |
| construction_vehicle |  0.000        |
| pedestrian        |  0.405        |
| motorcycle        |  0.226        |
| bicycle           |  0.000        |
| traffic_cone      |  0.000        |
| barrier           |  0.000        |

**mAP**
| mAP |
|-----|
|  0.120 |

**TP Errors (Translational, Scale, Orientation, Velocity, Attribute)**
| Label             | Trans Err | Scale Err | Orient Err | Vel Err | Attr Err |
|-------------------|-----------|-----------|------------|---------|----------|
| car               |  0.526     |  0.154     |  0.101      |  1.685   |  1.000    |
| truck             |  1.000     |  1.000     |  1.000      |  1.000   |  1.000    |
| bus               |  1.000     |  1.000     |  1.000      |  1.000   |  1.000    |
| trailer           |  1.000     |  1.000     |  1.000      |  1.000   |  1.000    |
| construction_vehicle |  1.000     |  1.000     |  1.000      |  1.000   |  1.000    |
| pedestrian        |  0.707     |  0.273     |  0.776      |  0.925   |  1.000    |
| motorcycle        |  0.600     |  0.207     |  0.265      |  4.088   |  1.000    |
| bicycle           |  1.000     |  1.000     |  1.000      |  1.000   |  1.000    |
| traffic_cone      |  1.000     |  1.000     |  nan      |  nan   |  nan    |
| barrier           |  1.000     |  1.000     |  1.000      |  nan   |  nan    |

**Average TP Errors Across All Labels (excluding NaNs)**
| Trans Err | Scale Err | Orient Err | Vel Err | Attr Err |
|-----------|-----------|------------|---------|----------|
|  0.883     |  0.763     |  0.793      |  1.462   |  1.000    |

**TP Scores (Translational, Scale, Orientation, Velocity, Attribute)**
| Metric            | TP Score   |
|-------------------|------------|
| trans_err         |  0.117      |
| scale_err         |  0.237      |
| orient_err        |  0.207      |
| vel_err           |  0.000      |
| attr_err          |  0.000      |

**ND Score**
| ND Score  |
|-----------|
|  0.116     |

**Evaluation Time**
| Evaluation Time (s) |
|---------------------|
|  647.824              |