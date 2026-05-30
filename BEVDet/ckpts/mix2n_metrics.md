**Label APS**
| Label             | 0.5      | 1.0      | 2.0      | 4.0      |
|-------------------|----------|----------|----------|----------|
| car               |  0.178  |  0.485  |  0.707  |  0.797  |
| truck             |  0.000  |  0.000  |  0.000  |  0.000  |
| bus               |  0.000  |  0.000  |  0.000  |  0.000  |
| trailer           |  0.000  |  0.000  |  0.000  |  0.000  |
| construction_vehicle |  0.000  |  0.000  |  0.000  |  0.000  |
| pedestrian        |  0.131  |  0.283  |  0.403  |  0.466  |
| motorcycle        |  0.062  |  0.188  |  0.279  |  0.315  |
| bicycle           |  0.000  |  0.000  |  0.000  |  0.000  |
| traffic_cone      |  0.000  |  0.000  |  0.000  |  0.000  |
| barrier           |  0.000  |  0.000  |  0.000  |  0.000  |

**Mean Dist APS**
| Label             | Mean Dist AP |
|-------------------|--------------|
| car               |  0.542        |
| truck             |  0.000        |
| bus               |  0.000        |
| trailer           |  0.000        |
| construction_vehicle |  0.000        |
| pedestrian        |  0.321        |
| motorcycle        |  0.211        |
| bicycle           |  0.000        |
| traffic_cone      |  0.000        |
| barrier           |  0.000        |

**mAP**
| mAP |
|-----|
|  0.107 |

**TP Errors (Translational, Scale, Orientation, Velocity, Attribute)**
| Label             | Trans Err | Scale Err | Orient Err | Vel Err | Attr Err |
|-------------------|-----------|-----------|------------|---------|----------|
| car               |  0.527     |  0.167     |  0.101      |  1.776   |  0.466    |
| truck             |  1.000     |  1.000     |  1.000      |  1.000   |  1.000    |
| bus               |  1.000     |  1.000     |  1.000      |  1.000   |  1.000    |
| trailer           |  1.000     |  1.000     |  1.000      |  1.000   |  1.000    |
| construction_vehicle |  1.000     |  1.000     |  1.000      |  1.000   |  1.000    |
| pedestrian        |  0.759     |  0.295     |  0.749      |  0.842   |  0.735    |
| motorcycle        |  0.734     |  0.281     |  0.998      |  1.068   |  0.289    |
| bicycle           |  1.000     |  1.000     |  1.000      |  1.000   |  1.000    |
| traffic_cone      |  1.000     |  1.000     |  nan      |  nan   |  nan    |
| barrier           |  1.000     |  1.000     |  1.000      |  nan   |  nan    |

**Average TP Errors Across All Labels (excluding NaNs)**
| Trans Err | Scale Err | Orient Err | Vel Err | Attr Err |
|-----------|-----------|------------|---------|----------|
|  0.902     |  0.774     |  0.872      |  1.086   |  0.811    |

**TP Scores (Translational, Scale, Orientation, Velocity, Attribute)**
| Metric            | TP Score   |
|-------------------|------------|
| trans_err         |  0.098      |
| scale_err         |  0.226      |
| orient_err        |  0.128      |
| vel_err           |  0.000      |
| attr_err          |  0.189      |

**ND Score**
| ND Score  |
|-----------|
|  0.118     |

**Evaluation Time**
| Evaluation Time (s) |
|---------------------|
|  89.158              |