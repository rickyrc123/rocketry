# Lesson 1: Rocket Dynamics and Trajectory Analysis

## Learning Objectives
By the end of this lesson, you will be able to:
1. Derive and apply 2D equations of motion for rocket flight (6-DOF dynamics)
2. Understand and implement gravity-turn trajectories
3. Apply numerical integration methods (Euler and Runge-Kutta 4th order) to solve differential equations
4. Simulate basic rocket trajectories using Python

## 1. Introduction to Rocket Dynamics

Rocket flight is a complex problem involving multiple forces, changing mass, and non-linear dynamics. We'll start with simplified 2D motion and build up to more complex scenarios.

### Key Assumptions for 2D Flight
- Rocket moves in a vertical plane (no lateral wind or Earth rotation effects initially)
- Earth is flat (valid for short-range trajectories)
- Atmospheric drag is simplified or neglected initially
- Rocket is treated as a point mass

## 2. Six Degrees of Freedom (6-DOF) Dynamics

### 2.1 Coordinate Systems

For rocket analysis, we typically use two coordinate systems:

**Earth-Fixed (Inertial) Frame:**
- x-axis: horizontal, positive eastward
- y-axis: vertical, positive upward
- Origin at launch point

**Body-Fixed Frame:**
- x_b-axis: along rocket centerline (positive forward)
- y_b-axis: perpendicular to centerline
- Rotated by angle θ from earth-fixed frame

### 2.2 Forces Acting on a Rocket

1. **Thrust Force (T)**: Propulsive force along rocket axis
2. **Weight (W)**: mg, always pointing downward
3. **Drag (D)**: Opposite to velocity direction
4. **Normal Forces**: Aerodynamic forces perpendicular to velocity

### 2.3 The Six Degrees of Freedom

For a rigid body in 3D space:
- **3 Translational DOFs**: position (x, y, z)
- **3 Rotational DOFs**: attitude (roll φ, pitch θ, yaw ψ)

For 2D flight, we reduce to:
- **2 Translational DOFs**: position (x, y)
- **1 Rotational DOF**: pitch angle θ

### 2.4 Equations of Motion (2D Case)

The fundamental equations come from Newton's second law:

**Force Equations:**
```
∑F_x = ma_x
∑F_y = ma_y
```

**Moment Equation:**
```
∑M = Iα
```

Where:
- m = rocket mass (time-varying due to fuel consumption)
- I = moment of inertia about center of mass
- α = angular acceleration

### 2.5 Detailed 2D Equations

**Translational Equations:**
```
m(dv_x/dt) = T cos(θ) - D cos(γ)
m(dv_y/dt) = T sin(θ) - mg - D sin(γ)
```

Where:
- v_x, v_y = velocity components in earth-fixed frame
- θ = pitch angle (angle between rocket axis and horizontal)
- γ = flight path angle (angle between velocity vector and horizontal)
- T = thrust magnitude
- D = drag magnitude
- g = gravitational acceleration

**Kinematic Equations:**
```
dx/dt = v_x
dy/dt = v_y
```

**Mass Equation:**
```
dm/dt = -ṁ = -T/(I_sp × g₀)
```

Where:
- ṁ = mass flow rate
- I_sp = specific impulse
- g₀ = standard gravity (9.81 m/s²)

**Angular Equation (simplified):**
```
dθ/dt = q
```

Where q is the pitch rate (for gravity turn, this will be determined by the trajectory requirements).

## 3. Gravity-Turn Trajectories

### 3.1 Concept of Gravity Turn

A gravity turn is a trajectory optimization technique where:
1. Rocket launches vertically
2. After reaching sufficient altitude/velocity, rocket tips over slightly
3. Gravitational torque gradually turns the rocket toward horizontal
4. No active steering required - gravity does the work!

### 3.2 Advantages of Gravity Turn
- Minimizes gravity losses
- Reduces structural loads (no large steering forces)
- Efficient trajectory for reaching orbit
- Simple to implement

### 3.3 Mathematical Description

For a gravity turn, the flight path angle γ and pitch angle θ are related by the constraint that aerodynamic angle of attack is zero (α = θ - γ = 0), so:

```
θ = γ
```

The flight path angle evolves according to:
```
dγ/dt = -(g cos γ)/V + (T sin α)/(mV)
```

For zero angle of attack (α = 0):
```
dγ/dt = -(g cos γ)/V
```

### 3.4 Gravity Turn Implementation

The gravity turn can be implemented by:
1. **Pitch-over maneuver**: Small initial pitch at low altitude
2. **Free flight**: Let gravity turn the rocket
3. **Steering law**: θ = γ (rocket points along velocity vector)

## 4. Numerical Integration Methods

Since rocket equations of motion are differential equations that rarely have analytical solutions, we need numerical methods.

### 4.1 Euler's Method (First-Order)

The simplest numerical integration method:

```
y_(n+1) = y_n + h × f(t_n, y_n)
```

Where:
- h = time step
- f(t, y) = dy/dt (the derivative function)
- y_n = value at time step n

**Advantages:** Simple, fast
**Disadvantages:** Low accuracy, can be unstable

### 4.2 Runge-Kutta 4th Order (RK4)

A much more accurate method:

```
k₁ = h × f(t_n, y_n)
k₂ = h × f(t_n + h/2, y_n + k₁/2)
k₃ = h × f(t_n + h/2, y_n + k₂/2)
k₄ = h × f(t_n + h, y_n + k₃)

y_(n+1) = y_n + (k₁ + 2k₂ + 2k₃ + k₄)/6
```

**Advantages:** Much higher accuracy (4th order vs 1st order)
**Disadvantages:** More computationally expensive (4 function evaluations per step)

### 4.3 Choosing Integration Parameters

**Time Step Selection:**
- Too large: Loss of accuracy, potential instability
- Too small: Unnecessary computation, accumulation of round-off errors
- Rule of thumb: Choose h such that important dynamics are resolved

**Stability Considerations:**
- Explicit methods (like Euler, RK4) have stability limits
- For stiff systems, implicit methods may be needed

## 5. Implementation Strategy

To simulate rocket trajectories:

1. **Define State Vector:**
   ```
   state = [x, y, v_x, v_y, m, θ]
   ```

2. **Define Derivative Function:**
   ```python
   def derivatives(t, state, rocket_params):
       # Extract state variables
       # Calculate forces
       # Return derivatives
   ```

3. **Choose Integration Method:**
   - Use RK4 for accuracy
   - Use appropriate time step

4. **Implement Trajectory Constraints:**
   - Gravity turn logic
   - Engine cutoff conditions
   - Atmospheric effects

## 6. Example Problem

**Problem:** Simulate the trajectory of a rocket with the following parameters:
- Initial mass: 1000 kg
- Fuel mass: 800 kg
- Thrust: 15,000 N
- Specific impulse: 250 s
- Launch angle: 90° (vertical)
- Pitch-over at t = 10 s to 85°

**Solution approach:**
1. Set up equations of motion
2. Implement gravity turn after pitch-over
3. Use RK4 integration with h = 0.1 s
4. Simulate until fuel depletion or impact

## 7. Advanced Considerations

### 7.1 Atmospheric Effects
- Drag force: D = ½ρV²C_D A
- Density variation with altitude: ρ(h)
- Wind effects

### 7.2 Earth Curvature
For long-range trajectories:
- Spherical Earth model
- Gravitational variation with altitude
- Coriolis effects

### 7.3 Multi-Stage Rockets
- Stage separation logic
- Changing vehicle parameters
- Optimal staging

### 7.4 Trajectory Optimization
- Minimize ΔV requirements
- Constrained optimization
- Calculus of variations

## 8. Summary

Key concepts covered:
1. **6-DOF dynamics** provide the framework for rocket motion analysis
2. **2D equations of motion** capture the essential physics for planar flight
3. **Gravity turns** offer an efficient trajectory for orbital insertion
4. **Numerical integration** (Euler, RK4) enables solution of complex differential equations
5. **Implementation considerations** include time step selection and stability

## 9. Practice Problems

1. Derive the equations of motion for a rocket in a uniform gravitational field with drag proportional to V².

2. Calculate the trajectory of a rocket using Euler's method with h = 1.0 s and compare with h = 0.1 s.

3. Implement a gravity turn starting at 1000 m altitude and compare with a vertical trajectory.

4. Analyze the effect of specific impulse on maximum altitude reached.

## 10. References

1. Sutton, G. P., & Biblarz, O. (2016). *Rocket Propulsion Elements*. John Wiley & Sons.
2. Turner, M. J. L. (2009). *Rocket and Spacecraft Propulsion*. Springer.
3. Wiesel, W. E. (2010). *Spaceflight Dynamics*. Apogee Books.
4. Prussing, J. E., & Conway, B. A. (2012). *Orbital Mechanics*. Oxford University Press.

---

*This lesson provides the theoretical foundation for rocket trajectory analysis. The accompanying Python exercises will help you implement these concepts and gain practical experience with numerical simulation.*
