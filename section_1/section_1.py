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
        self.mdot = _______________
        
        # TODO 2: Calculate engine burn time
        # Hint: burn_time = fuel_mass / mdot
        self.burn_time = _______________
        
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
        
        return _______________
    
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
        drag_magnitude = _______________
        
        # TODO 5: Calculate drag direction (opposite to velocity)
        drag_direction = _______________
        
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
        current_thrust = _______________
        
        # Calculate forces
        # TODO 7: Calculate thrust force components
        # Thrust acts along the rocket body axis (angle theta)
        thrust_x = _______________
        thrust_y = _______________
        
        # TODO 8: Calculate gravitational force components
        # Weight always acts downward
        weight_x = _______________
        weight_y = _______________
        
        # Calculate drag forces
        velocity = np.array([vx, vy])
        drag = self.drag_force(velocity, y)
        
        # TODO 9: Apply Newton's second law to get accelerations
        # F = ma, so a = F/m
        ax = _______________
        ay = _______________
        
        # TODO 10: Calculate mass change rate
        # dm/dt = -mdot when engine is on, 0 when off
        dmdt = _______________
        
        # TODO 11: Implement gravity turn logic
        # For gravity turn: theta should follow the flight path angle
        # Flight path angle: gamma = arctan(vy / vx)
        # For simplicity, make theta follow gamma with some delay
        if engine_on and t > 10:  # Start gravity turn after 10 seconds
            flight_path_angle = _______________
            dtheta_dt = 0.1 * (flight_path_angle - theta)  # Simple proportional control
        else:
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
        state = np.array([_______________])
        
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
            state = _______________
        
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
            k1 = _______________
            k2 = _______________
            k3 = _______________
            k4 = _______________
            
            # TODO 15: Combine k coefficients for final update
            state = _______________
        
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
    vel_rk4 = _______________
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
    pitch_rk4 = _______________
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
    # TODO 21: Create two simulations - one with gravity turn, one without
    
    # Rocket parameters
    rocket_params = {
        'initial_mass': 1000,  # kg
        'fuel_mass': 800,      # kg
        'thrust': 15000,       # N
        'specific_impulse': 250, # s
        'drag_coefficient': 0.3,
        'reference_area': 1.0   # m^2
    }
    
    # Simulation with gravity turn
    sim_gravity = RocketSimulation(rocket_params)
    time_g, states_g = sim_gravity.rk4_integration(0.1, 200)
    
    # TODO 22: Create a modified simulation for vertical flight only
    # Modify the equations_of_motion to keep theta = pi/2 always
    # This will require a new class or method modification
    
    print("Gravity Turn Analysis:")
    print(f"Maximum altitude with gravity turn: {np.max(states_g[:, 1]):.1f} m")
    print(f"Maximum downrange with gravity turn: {np.max(states_g[:, 0]):.1f} m")

def main():
    """
    Main function to run the rocket simulation
    """
    print("Rocket Dynamics Simulation")
    print("=" * 30)
    
    # TODO 23: Define rocket parameters
    rocket_params = {
        'initial_mass': _______________,     # kg
        'fuel_mass': _______________,        # kg  
        'thrust': _______________,           # N
        'specific_impulse': _______________,  # s
        'drag_coefficient': 0.3,
        'reference_area': 1.0                # m^2
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
    time_euler, states_euler = _______________
    time_rk4, states_rk4 = _______________
    
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
    alt_diff = _______________
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