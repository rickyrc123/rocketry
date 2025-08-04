"""
Interactive Rocket Dynamics Lesson - Fill in the Blanks
=====================================================

This Python script implements the concepts from the Rocket Dynamics lesson.
Fill in the blanks (marked with # TODO) to complete the implementation.

Topics covered:
1. 2D equations of motion (6-DOF dynamics)
2. Gravity-turn trajectories  
3. Numerical integration (Euler and RK4 methods)

Instructions:
- Look for # TODO comments and fill in the missing code
- Test your implementation by running the script
- Compare your results with the answer key
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

class RocketSimulation:
    """
    A class to simulate 2D rocket trajectories with gravity turn
    """
    
    def __init__(self, rocket_params: dict):
        """
        Initialize rocket simulation with parameters
        
        Args:
            rocket_params: Dictionary containing rocket specifications
        """
        self.m0 = rocket_params['initial_mass']  # kg
        self.fuel_mass = rocket_params['fuel_mass']  # kg
        self.thrust = rocket_params['thrust']  # N
        self.isp = rocket_params['specific_impulse']  # s
        self.g0 = 9.81  # m/s^2
        self.drag_coeff = rocket_params.get('drag_coefficient', 0.0)
        self.reference_area = rocket_params.get('reference_area', 1.0)
        
        # TODO 1: Calculate the mass flow rate using the rocket equation
        # Hint: mdot = thrust / (isp * g0)
        self.mdot = self.thrust / (self.isp * 9.81)#_______________
        
        # TODO 2: Calculate engine burn time
        # Hint: burn_time = fuel_mass / mdot
        self.burn_time = self.fuel_mass / self.mdot #_______________
        
    def atmospheric_density(self, altitude: float) -> float:
        """
        Calculate atmospheric density using exponential model
        
        Args:
            altitude: Altitude in meters
            
        Returns:
            Atmospheric density in kg/m^3
        """
        # TODO 3: Implement exponential atmosphere model
        # Use: rho = rho0 * exp(-altitude / scale_height)
        # where rho0 = 1.225 kg/m^3, scale_height = 8000 m
        rho0 = 1.225
        scale_height = 8000
        
        return rho0 * np.exp(-altitude / scale_height)#_______________
    
    def drag_force(self, velocity: np.ndarray, altitude: float) -> np.ndarray:
        """
        Calculate drag force vector
        
        Args:
            velocity: Velocity vector [vx, vy]
            altitude: Current altitude
            
        Returns:
            Drag force vector [Dx, Dy]
        """
        speed = np.linalg.norm(velocity)
        if speed == 0:
            return np.array([0.0, 0.0])
            
        rho = self.atmospheric_density(altitude)
        
        # TODO 4: Calculate drag magnitude
        # Use: D = 0.5 * rho * V^2 * Cd * A
        drag_magnitude = 0.5 * rho * speed**2 * self.drag_coeff * self.reference_area#_______________
        
        # TODO 5: Calculate drag direction (opposite to velocity)
        drag_direction = -velocity/speed #_______________
        
        return drag_magnitude * drag_direction
    
    def equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute derivatives for the rocket equations of motion
        
        Args:
            t: Current time
            state: State vector [x, y, vx, vy, mass, theta]
            
        Returns:
            Derivatives [dx/dt, dy/dt, dvx/dt, dvy/dt, dm/dt, dtheta/dt]
        """
        x, y, vx, vy, mass, theta = state
        
        # Check if engine is still burning
        engine_on = t < self.burn_time
        
        # TODO 6: Calculate current thrust (0 if engine off)
        current_thrust = self.thrust if engine_on else 0.0 #_______________
        
        # Calculate forces
        # TODO 7: Calculate thrust force components
        # Thrust acts along the rocket body axis (angle theta)
        thrust_x = current_thrust * np.cos(theta) #_______________
        thrust_y = current_thrust * np.sin(theta) #_______________
        
        # TODO 8: Calculate gravitational force components
        # Weight always acts downward
        weight_x = 0.0#_______________
        weight_y = -mass * self.g0 #_______________
        
        # Calculate drag forces
        velocity = np.array([vx, vy])
        drag = self.drag_force(velocity, y)
        
        # TODO 9: Apply Newton's second law to get accelerations
        # F = ma, so a = F/m
        ax = (thrust_x + weight_x - drag[0]) / mass #_______________
        ay = (thrust_y + weight_y - drag[1]) / mass #_______________
        
        # TODO 10: Calculate mass change rate
        # dm/dt = -mdot when engine is on, 0 when off
        dmdt = -self.mdot if engine_on else 0.0 #_______________
        
        # TODO 11: Implement gravity turn logic
        if engine_on and t > 10:  # Start gravity turn after 10 seconds
            if t <= 15:  # Initial pitch-over maneuver (10-15 seconds)
                target_theta = np.pi/2 - 0.1 * (t - 10)  # Gradually pitch from 90° to 85°
                dtheta_dt = 0.5 * (target_theta - theta)  # Quick response to initiate turn
            else:  # Follow flight path angle after initial maneuver
                if abs(vx) > 1.0:  # Only when we have horizontal velocity
                    flight_path_angle = np.arctan2(vy, vx)
                    dtheta_dt = 0.1 * (flight_path_angle - theta)
                else:
                    dtheta_dt = 0.0
        else:
            dtheta_dt = 0.0
        
        return np.array([vx, vy, ax, ay, dmdt, dtheta_dt])
    
    def equations_of_motion_vertical(self, t: float, state: np.ndarray) -> np.ndarray:
        x, y, vx, vy, mass, theta = state

        theta = np.pi/2

        engine_on = t < self.burn_time
        current_thrust = self.thrust if engine_on else 0.0

        thrust_x = 0.0
        thrust_y = current_thrust

        weight_x = 0.0
        weight_y = -mass * self.g0

        velocity = np.array([vx, vy])
        drag = self.drag_force(velocity, y)

        ax = (thrust_x + weight_x - drag[0]) / mass
        ay = (thrust_y + weight_y - drag[1]) / mass

        dmdt = -self.mdot if engine_on else 0.0

        dtheta_dt = 0.0

        return np.array([vx, vy, ax, ay, dmdt, dtheta_dt])

    def euler_integration(self, dt: float, t_end: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate equations of motion using Euler's method
        
        Args:
            dt: Time step
            t_end: End time
            
        Returns:
            time_array, state_history
        """
        # Initial conditions
        # TODO 12: Set initial state vector
        # [x, y, vx, vy, mass, theta] = [0, 0, 0, 0, initial_mass, pi/2]
        state = np.array([0.0, 0.0, 0.0, 0.0, self.m0, np.pi/2]) 
        
        # Initialize arrays
        time_points = np.arange(0, t_end + dt, dt)
        state_history = np.zeros((len(time_points), 6))
        
        for i, t in enumerate(time_points):
            state_history[i] = state
            
            # Check if rocket has hit the ground
            if state[1] < 0 and i > 0:  # y < 0 (below ground)
                return time_points[:i], state_history[:i]
            
            # TODO 13: Implement Euler's method
            # state_new = state_old + dt * derivatives
            derivatives = self.equations_of_motion(t, state)
            state = state + dt * derivatives #_______________
            
            # Check for numerical instability
            if np.any(np.isnan(state)) or np.any(np.abs(state) > 1e10):
                print(f"Numerical instability detected at t={t:.1f}s")
                return time_points[:i], state_history[:i]
        
        return time_points, state_history
    
    def rk4_integration(self, dt: float, t_end: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate equations of motion using 4th-order Runge-Kutta method
        
        Args:
            dt: Time step
            t_end: End time
            
        Returns:
            time_array, state_history
        """
        # Initial conditions
        state = np.array([0.0, 0.0, 0.0, 0.0, self.m0, np.pi/2])
        
        # Initialize arrays
        time_points = np.arange(0, t_end + dt, dt)
        state_history = np.zeros((len(time_points), 6))
        
        for i, t in enumerate(time_points):
            state_history[i] = state
            
            # Check if rocket has hit the ground
            if state[1] < 0 and i > 0:
                return time_points[:i], state_history[:i]
            
            # TODO 14: Implement RK4 method
            # Calculate k1, k2, k3, k4 coefficients
            k1 = dt * self.equations_of_motion(t, state) #_______________
            k2 = dt * self.equations_of_motion(t + dt/2, state + k1/2) #_______________
            k3 = dt * self.equations_of_motion(t + dt/2, state + k2/2) #_______________
            k4 = dt * self.equations_of_motion(t + dt, state + k3) #_______________
            
            # TODO 15: Combine k coefficients for final update
            state = state + (k1 + 2*k2 + 2*k3 + k4)/6 #_______________
            
            # Check for numerical instability
            if np.any(np.isnan(state)) or np.any(np.abs(state) > 1e10):
                print(f"Numerical instability detected at t={t:.1f}s")
                return time_points[:i], state_history[:i]
        
        return time_points, state_history

def plot_trajectory(time_euler, states_euler, time_rk4, states_rk4):
    """
    Plot trajectory comparison between Euler and RK4 methods
    """
    # TODO 16: Create subplot for trajectory comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # TODO 17: Plot altitude vs time
    axes[0, 0].plot(time_euler, states_euler[:, 1], 'r--', label='Euler', linewidth=2)
    axes[0, 0].plot(time_rk4, states_rk4[:, 1], 'b-', label='RK4', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Altitude (m)')
    axes[0, 0].set_title('Altitude vs Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # TODO 18: Plot velocity magnitude vs time
    vel_euler = np.sqrt(states_euler[:, 2]**2 + states_euler[:, 3]**2)
    vel_rk4 = np.sqrt(states_rk4[:, 2]**2 + states_rk4[:, 3]**2) #_______________
    axes[0, 1].plot(time_euler, vel_euler, 'r--', label='Euler', linewidth=2)
    axes[0, 1].plot(time_rk4, vel_rk4, 'b-', label='RK4', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Velocity vs Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # TODO 19: Plot trajectory in x-y plane
    axes[1, 0].plot(states_euler[:, 0], states_euler[:, 1], 'r--', label='Euler', linewidth=2)
    axes[1, 0].plot(states_rk4[:, 0], states_rk4[:, 1], 'b-', label='RK4', linewidth=2)
    axes[1, 0].set_xlabel('Downrange (m)')
    axes[1, 0].set_ylabel('Altitude (m)')
    axes[1, 0].set_title('Trajectory')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # TODO 20: Plot pitch angle vs time
    pitch_euler = np.degrees(states_euler[:, 5])
    pitch_rk4 = np.degrees(states_rk4[:, 5]) #_______________
    axes[1, 1].plot(time_euler, pitch_euler, 'r--', label='Euler', linewidth=2)
    axes[1, 1].plot(time_rk4, pitch_rk4, 'b-', label='RK4', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Pitch Angle (degrees)')
    axes[1, 1].set_title('Pitch Angle vs Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_gravity_turn_effectiveness():
    """
    Compare vertical flight vs gravity turn trajectory
    """
    # Rocket parameters
    rocket_params = {
        'initial_mass': 1000,  # kg
        'fuel_mass': 800,      # kg
        'thrust': 10000,       # N
        'specific_impulse': 220, # s
        'drag_coefficient': 0.6,
        'reference_area': 2.5   # m^2
    }
    
    print(f"Analysis function using: Thrust={rocket_params['thrust']}N, ISP={rocket_params['specific_impulse']}s")
    
    # Simulation with gravity turn
    sim_gravity = RocketSimulation(rocket_params)
    print(f"Gravity sim - Burn time: {sim_gravity.burn_time:.1f}s, mdot: {sim_gravity.mdot:.3f}kg/s")
    
    # Change from 0.1 to 0.01 seconds
    time_g, states_g = sim_gravity.rk4_integration(0.01, 300)  # Smaller dt, longer time
    print(f"Debug - Gravity turn simulation: {len(time_g)} points, max alt: {np.max(states_g[:, 1]) if len(states_g) > 0 else 0:.1f} m")

    # Check if arrays are empty
    if len(states_g) == 0:
        print("ERROR: Gravity turn simulation returned empty array!")
        return

    sim_vertical = RocketSimulation(rocket_params)
    original_method = sim_vertical.equations_of_motion
    sim_vertical.equations_of_motion = sim_vertical.equations_of_motion_vertical
    time_v, states_v = sim_vertical.rk4_integration(0.01, 300)  # Smaller dt, longer time
    print(f"Debug - Vertical simulation: {len(time_v)} points, max alt: {np.max(states_v[:, 1]):.1f} m")
    sim_vertical.equations_of_motion = original_method

    
    # TODO 22: Create a modified simulation for vertical flight only
    # Modify the equations_of_motion to keep theta = pi/2 always
    # This will require a new class or method modification
    
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

def main():
    """
    Main function to run the rocket simulation
    """
    print("Rocket Dynamics Simulation")
    print("=" * 30)
    
    # TODO 23: Define rocket parameters
    rocket_params = {
        'initial_mass': 1000,  # kg
        'fuel_mass': 800,      # kg
        'thrust': 10000,       # N
        'specific_impulse': 220, # s
        'drag_coefficient': 0.6,
        'reference_area': 2.5   # m^2
    }
    
    # Create simulation object
    sim = RocketSimulation(rocket_params)
    
    # Print rocket characteristics
    print(f"Initial mass: {sim.m0} kg")
    print(f"Fuel mass: {sim.fuel_mass} kg")
    print(f"Thrust: {sim.thrust} N")
    print(f"Specific impulse: {sim.isp} s")
    print(f"Burn time: {sim.burn_time:.1f} s")
    print()
    
    # TODO 24: Run simulations with both integration methods
    print("Running simulations...")
    time_euler, states_euler = sim.euler_integration(0.1, 200)
    time_rk4, states_rk4 = sim.rk4_integration(0.05, 200)
    
    # Display results
    print("\nResults:")
    print("-" * 20)
    print("Euler Method:")
    print(f"  Maximum altitude: {np.max(states_euler[:, 1]):.1f} m")
    print(f"  Maximum velocity: {np.max(np.sqrt(states_euler[:, 2]**2 + states_euler[:, 3]**2)):.1f} m/s")
    
    print("\nRK4 Method:")
    print(f"  Maximum altitude: {np.max(states_rk4[:, 1]):.1f} m")
    print(f"  Maximum velocity: {np.max(np.sqrt(states_rk4[:, 2]**2 + states_rk4[:, 3]**2)):.1f} m/s")
    
    # TODO 25: Calculate and display the difference between methods
    alt_diff = np.max(states_rk4[:, 1]) - np.max(states_euler[:, 1])
    print(f"\nDifference in max altitude: {alt_diff:.1f} m")
    
    # Plot results
    plot_trajectory(time_euler, states_euler, time_rk4, states_rk4)
    
    # Analyze gravity turn
    analyze_gravity_turn_effectiveness()

if __name__ == "__main__":
    main()

# TODO 26: Add error checking and input validation
# TODO 27: Implement multi-stage rocket capability  
# TODO 28: Add atmospheric wind effects
# TODO 29: Include Earth curvature for long-range trajectories
# TODO 30: Optimize trajectory for maximum range or altitude