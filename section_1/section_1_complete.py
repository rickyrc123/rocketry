"""
Complete Working Solution - Rocket Dynamics Simulation
====================================================

This is the complete working version with all TODOs filled in.
Use this for reference or to verify your implementation.

Topics covered:
1. 2D equations of motion (6-DOF dynamics)
2. Gravity-turn trajectories  
3. Numerical integration (Euler and RK4 methods)
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
        
        # Calculate the mass flow rate using the rocket equation
        self.mdot = self.thrust / (self.isp * self.g0)
        
        # Calculate engine burn time
        self.burn_time = self.fuel_mass / self.mdot
        
    def atmospheric_density(self, altitude: float) -> float:
        """
        Calculate atmospheric density using exponential model
        
        Args:
            altitude: Altitude in meters
            
        Returns:
            Atmospheric density in kg/m^3
        """
        rho0 = 1.225
        scale_height = 8000
        
        return rho0 * np.exp(-altitude / scale_height)
    
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
        
        # Calculate drag magnitude
        drag_magnitude = 0.5 * rho * speed**2 * self.drag_coeff * self.reference_area
        
        # Calculate drag direction (opposite to velocity)
        drag_direction = -velocity / speed
        
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
        
        # Calculate current thrust (0 if engine off)
        current_thrust = self.thrust if engine_on else 0.0
        
        # Calculate forces
        # Calculate thrust force components
        thrust_x = current_thrust * np.cos(theta)
        thrust_y = current_thrust * np.sin(theta)
        
        # Calculate gravitational force components
        weight_x = 0.0
        weight_y = -mass * self.g0
        
        # Calculate drag forces
        velocity = np.array([vx, vy])
        drag = self.drag_force(velocity, y)
        
        # Apply Newton's second law to get accelerations
        ax = (thrust_x + weight_x - drag[0]) / mass
        ay = (thrust_y + weight_y - drag[1]) / mass
        
        # Calculate mass change rate
        dmdt = -self.mdot if engine_on else 0.0
        
        # Implement gravity turn logic
        if engine_on and t > 10:  # Start gravity turn after 10 seconds
            flight_path_angle = np.arctan2(vy, vx)
            dtheta_dt = 0.1 * (flight_path_angle - theta)  # Simple proportional control
        else:
            dtheta_dt = 0.0
        
        return np.array([vx, vy, ax, ay, dmdt, dtheta_dt])
    
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
        state = np.array([0.0, 0.0, 0.0, 0.0, self.m0, np.pi/2])
        
        # Initialize arrays
        time_points = np.arange(0, t_end + dt, dt)
        state_history = np.zeros((len(time_points), 6))
        
        for i, t in enumerate(time_points):
            state_history[i] = state
            
            # Check if rocket has hit the ground
            if state[1] < 0 and i > 0:  # y < 0 (below ground)
                return time_points[:i], state_history[:i]
            
            # Implement Euler's method
            derivatives = self.equations_of_motion(t, state)
            state = state + dt * derivatives
        
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
            
            # Implement RK4 method
            k1 = dt * self.equations_of_motion(t, state)
            k2 = dt * self.equations_of_motion(t + dt/2, state + k1/2)
            k3 = dt * self.equations_of_motion(t + dt/2, state + k2/2)
            k4 = dt * self.equations_of_motion(t + dt, state + k3)
            
            # Combine k coefficients for final update
            state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return time_points, state_history

def plot_trajectory(time_euler, states_euler, time_rk4, states_rk4):
    """
    Plot trajectory comparison between Euler and RK4 methods
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot altitude vs time
    axes[0, 0].plot(time_euler, states_euler[:, 1], 'r--', label='Euler', linewidth=2)
    axes[0, 0].plot(time_rk4, states_rk4[:, 1], 'b-', label='RK4', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Altitude (m)')
    axes[0, 0].set_title('Altitude vs Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot velocity magnitude vs time
    vel_euler = np.sqrt(states_euler[:, 2]**2 + states_euler[:, 3]**2)
    vel_rk4 = np.sqrt(states_rk4[:, 2]**2 + states_rk4[:, 3]**2)
    axes[0, 1].plot(time_euler, vel_euler, 'r--', label='Euler', linewidth=2)
    axes[0, 1].plot(time_rk4, vel_rk4, 'b-', label='RK4', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Velocity vs Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot trajectory in x-y plane
    axes[1, 0].plot(states_euler[:, 0], states_euler[:, 1], 'r--', label='Euler', linewidth=2)
    axes[1, 0].plot(states_rk4[:, 0], states_rk4[:, 1], 'b-', label='RK4', linewidth=2)
    axes[1, 0].set_xlabel('Downrange (m)')
    axes[1, 0].set_ylabel('Altitude (m)')
    axes[1, 0].set_title('Trajectory')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot pitch angle vs time
    pitch_euler = np.degrees(states_euler[:, 5])
    pitch_rk4 = np.degrees(states_rk4[:, 5])
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

def main():
    """
    Main function to run the rocket simulation
    """
    print("Rocket Dynamics Simulation")
    print("=" * 30)
    
    # Define rocket parameters
    rocket_params = {
        'initial_mass': 1000,     # kg
        'fuel_mass': 800,         # kg  
        'thrust': 15000,          # N
        'specific_impulse': 250,  # s
        'drag_coefficient': 0.3,
        'reference_area': 1.0     # m^2
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
    
    # Run simulations with both integration methods
    print("Running simulations...")
    time_euler, states_euler = sim.euler_integration(0.1, 200)
    time_rk4, states_rk4 = sim.rk4_integration(0.1, 200)
    
    # Display results
    print("\nResults:")
    print("-" * 20)
    print("Euler Method:")
    print(f"  Maximum altitude: {np.max(states_euler[:, 1]):.1f} m")
    print(f"  Maximum velocity: {np.max(np.sqrt(states_euler[:, 2]**2 + states_euler[:, 3]**2)):.1f} m/s")
    
    print("\nRK4 Method:")
    print(f"  Maximum altitude: {np.max(states_rk4[:, 1]):.1f} m")
    print(f"  Maximum velocity: {np.max(np.sqrt(states_rk4[:, 2]**2 + states_rk4[:, 3]**2)):.1f} m/s")
    
    # Calculate and display the difference between methods
    alt_diff = np.max(states_rk4[:, 1]) - np.max(states_euler[:, 1])
    print(f"\nDifference in max altitude: {alt_diff:.1f} m")
    
    # Plot results
    plot_trajectory(time_euler, states_euler, time_rk4, states_rk4)
    
    # Analyze gravity turn
    analyze_gravity_turn_effectiveness()

if __name__ == "__main__":
    main()
