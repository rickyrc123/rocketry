# Section 1 Answer Key: Rocket Dynamics Implementation

This document contains all the answers for the fill-in-the-blank Python exercise in `section_1.py`.

## TODO Answers

### TODO 1: Calculate mass flow rate
```python
self.mdot = self.thrust / (self.isp * self.g0)
```
**Explanation:** The rocket equation relates thrust to mass flow rate through specific impulse: T = ṁ × Isp × g₀

### TODO 2: Calculate engine burn time
```python
self.burn_time = self.fuel_mass / self.mdot
```
**Explanation:** Burn time is simply the total fuel mass divided by the mass consumption rate.

### TODO 3: Implement exponential atmosphere model
```python
return rho0 * np.exp(-altitude / scale_height)
```
**Explanation:** The exponential atmosphere model assumes density decreases exponentially with altitude.

### TODO 4: Calculate drag magnitude
```python
drag_magnitude = 0.5 * rho * speed**2 * self.drag_coeff * self.reference_area
```
**Explanation:** Standard drag equation: D = ½ρV²CₐA

### TODO 5: Calculate drag direction
```python
drag_direction = -velocity / speed
```
**Explanation:** Drag always opposes motion, so it's in the opposite direction of the velocity vector.

### TODO 6: Calculate current thrust
```python
current_thrust = self.thrust if engine_on else 0.0
```
**Explanation:** Thrust is only active when the engine is burning (before fuel depletion).

### TODO 7: Calculate thrust force components
```python
thrust_x = current_thrust * np.cos(theta)
thrust_y = current_thrust * np.sin(theta)
```
**Explanation:** Thrust acts along the rocket body axis at angle θ from horizontal.

### TODO 8: Calculate gravitational force components
```python
weight_x = 0.0
weight_y = -mass * self.g0
```
**Explanation:** Gravity always acts vertically downward (negative y-direction).

### TODO 9: Apply Newton's second law
```python
ax = (thrust_x + weight_x - drag[0]) / mass
ay = (thrust_y + weight_y - drag[1]) / mass
```
**Explanation:** a = ΣF/m - sum all forces and divide by mass.

### TODO 10: Calculate mass change rate
```python
dmdt = -self.mdot if engine_on else 0.0
```
**Explanation:** Mass decreases at rate ṁ when engine is on, no change when off.

### TODO 11: Implement gravity turn logic
```python
flight_path_angle = np.arctan2(vy, vx)
```
**Explanation:** Flight path angle is the angle of the velocity vector: γ = arctan(vy/vx)

### TODO 12: Set initial state vector
```python
state = np.array([0.0, 0.0, 0.0, 0.0, self.m0, np.pi/2])
```
**Explanation:** Initial conditions: [x=0, y=0, vx=0, vy=0, mass=m₀, θ=90°]

### TODO 13: Implement Euler's method
```python
state = state + dt * derivatives
```
**Explanation:** Euler's method: y_{n+1} = y_n + h × f(t_n, y_n)

### TODO 14: Implement RK4 method - k coefficients
```python
k1 = dt * self.equations_of_motion(t, state)
k2 = dt * self.equations_of_motion(t + dt/2, state + k1/2)
k3 = dt * self.equations_of_motion(t + dt/2, state + k2/2)
k4 = dt * self.equations_of_motion(t + dt, state + k3)
```
**Explanation:** RK4 evaluates the derivative at four points within the time step.

### TODO 15: Combine k coefficients for final update
```python
state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
```
**Explanation:** RK4 combines the four slopes with weights 1:2:2:1.

### TODO 16: Create subplot for trajectory comparison
*This TODO is already implemented in the provided code structure.*

### TODO 17: Plot altitude vs time
*This TODO is already implemented in the provided code structure.*

### TODO 18: Plot velocity magnitude vs time
```python
vel_rk4 = np.sqrt(states_rk4[:, 2]**2 + states_rk4[:, 3]**2)
```
**Explanation:** Velocity magnitude is √(vx² + vy²)

### TODO 19: Plot trajectory in x-y plane
*This TODO is already implemented in the provided code structure.*

### TODO 20: Plot pitch angle vs time
```python
pitch_rk4 = np.degrees(states_rk4[:, 5])
```
**Explanation:** Convert radians to degrees for plotting.

### TODO 21: Create two simulations
*This requires modifying the class or creating a variant - see extended solution below.*

### TODO 22: Create modified simulation for vertical flight
*This requires a class modification - see extended solution below.*

### TODO 23: Define rocket parameters
```python
rocket_params = {
    'initial_mass': 1000,        # kg
    'fuel_mass': 800,           # kg  
    'thrust': 15000,            # N
    'specific_impulse': 250,     # s
    'drag_coefficient': 0.3,
    'reference_area': 1.0        # m^2
}
```

### TODO 24: Run simulations with both methods
```python
time_euler, states_euler = sim.euler_integration(0.1, 200)
time_rk4, states_rk4 = sim.rk4_integration(0.1, 200)
```

### TODO 25: Calculate difference between methods
```python
alt_diff = np.max(states_rk4[:, 1]) - np.max(states_euler[:, 1])
```

## Extended Solutions for Advanced TODOs

### For TODO 21-22: Vertical vs Gravity Turn Comparison

Add this method to the RocketSimulation class:

```python
def equations_of_motion_vertical(self, t: float, state: np.ndarray) -> np.ndarray:
    """
    Modified equations of motion for vertical flight only
    """
    x, y, vx, vy, mass, theta = state
    
    # Force theta to remain vertical (pi/2)
    theta = np.pi/2
    
    engine_on = t < self.burn_time
    current_thrust = self.thrust if engine_on else 0.0
    
    # Thrust components (always vertical)
    thrust_x = 0.0
    thrust_y = current_thrust
    
    # Weight components
    weight_x = 0.0
    weight_y = -mass * self.g0
    
    # Drag forces
    velocity = np.array([vx, vy])
    drag = self.drag_force(velocity, y)
    
    # Accelerations
    ax = (thrust_x + weight_x - drag[0]) / mass
    ay = (thrust_y + weight_y - drag[1]) / mass
    
    # Mass change
    dmdt = -self.mdot if engine_on else 0.0
    
    # Theta remains constant
    dtheta_dt = 0.0
    
    return np.array([vx, vy, ax, ay, dmdt, dtheta_dt])
```

Then update the `analyze_gravity_turn_effectiveness()` function:

```python
def analyze_gravity_turn_effectiveness():
    """
    Compare vertical flight vs gravity turn trajectory
    """
    rocket_params = {
        'initial_mass': 1000,
        'fuel_mass': 800,
        'thrust': 15000,
        'specific_impulse': 250,
        'drag_coefficient': 0.3,
        'reference_area': 1.0
    }
    
    # Simulation with gravity turn
    sim_gravity = RocketSimulation(rocket_params)
    time_g, states_g = sim_gravity.rk4_integration(0.1, 200)
    
    # Simulation with vertical trajectory
    # Create a modified version by temporarily replacing the equations
    sim_vertical = RocketSimulation(rocket_params)
    original_method = sim_vertical.equations_of_motion
    sim_vertical.equations_of_motion = sim_vertical.equations_of_motion_vertical
    time_v, states_v = sim_vertical.rk4_integration(0.1, 200)
    sim_vertical.equations_of_motion = original_method
    
    print("\nGravity Turn Analysis:")
    print("-" * 30)
    print(f"Gravity Turn Trajectory:")
    print(f"  Maximum altitude: {np.max(states_g[:, 1]):.1f} m")
    print(f"  Maximum downrange: {np.max(states_g[:, 0]):.1f} m")
    print(f"  Final velocity: {np.sqrt(states_g[-1, 2]**2 + states_g[-1, 3]**2):.1f} m/s")
    
    print(f"\nVertical Trajectory:")
    print(f"  Maximum altitude: {np.max(states_v[:, 1]):.1f} m")
    print(f"  Maximum downrange: {np.max(states_v[:, 0]):.1f} m") 
    print(f"  Final velocity: {np.sqrt(states_v[-1, 2]**2 + states_v[-1, 3]**2):.1f} m/s")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(states_g[:, 0], states_g[:, 1], 'b-', label='Gravity Turn', linewidth=2)
    plt.plot(states_v[:, 0], states_v[:, 1], 'r--', label='Vertical', linewidth=2)
    plt.xlabel('Downrange (m)')
    plt.ylabel('Altitude (m)')
    plt.title('Trajectory Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(time_g, states_g[:, 1], 'b-', label='Gravity Turn', linewidth=2)
    plt.plot(time_v, states_v[:, 1], 'r--', label='Vertical', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Altitude vs Time')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    vel_g = np.sqrt(states_g[:, 2]**2 + states_g[:, 3]**2)
    vel_v = np.sqrt(states_v[:, 2]**2 + states_v[:, 3]**2)
    plt.plot(time_g, vel_g, 'b-', label='Gravity Turn', linewidth=2)
    plt.plot(time_v, vel_v, 'r--', label='Vertical', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity vs Time')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(time_g, np.degrees(states_g[:, 5]), 'b-', label='Gravity Turn', linewidth=2)
    plt.plot(time_v, np.degrees(states_v[:, 5]), 'r--', label='Vertical', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Angle (degrees)')
    plt.title('Pitch Angle vs Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
```

## Key Learning Points

1. **Mass Flow Rate**: Directly related to thrust and specific impulse
2. **Atmospheric Model**: Exponential decay captures real atmosphere behavior
3. **Drag Force**: Quadratic dependence on velocity, always opposes motion
4. **Force Components**: Vector decomposition based on orientation angles
5. **Numerical Integration**: RK4 provides much higher accuracy than Euler
6. **Gravity Turn**: Naturally efficient trajectory that follows flight path angle
7. **State Vector**: Captures all necessary information about rocket state
8. **Time Step Selection**: Balance between accuracy and computation time

## Expected Results

When you run the completed simulation, you should see:

- **Maximum Altitude**: ~8000-12000 m (depending on drag model)
- **Burn Time**: ~200 seconds
- **RK4 vs Euler**: RK4 should give smoother, more accurate results
- **Gravity Turn**: Should show curved trajectory with increasing downrange
- **Pitch Angle**: Should gradually decrease from 90° to flight path angle

## Tips for Implementation

1. **Start Simple**: Get basic equations working before adding complexity
2. **Check Units**: Ensure all calculations use consistent units (SI)
3. **Debug Incrementally**: Test each TODO one at a time
4. **Validate Results**: Compare with known rocket performance data
5. **Plot Everything**: Visualization helps identify implementation errors

## Further Enhancements

After completing all TODOs, consider these extensions:
- Variable gravity with altitude
- Atmospheric wind effects
- Multi-stage rockets
- Trajectory optimization
- Monte Carlo analysis with uncertainties
- 3D flight with Earth rotation effects
