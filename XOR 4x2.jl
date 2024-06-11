using DifferentialEquations
using Plots

# Define the ODE function
function f!(dy, y, p, t)
    dy[1] = -y[1]
end

println("uff1")
# Initial condition
y0 = [1.0]

# Time span
tspan = (0.0, 5.0)

t_points = 0.0:0.1:5.0

# Define the ODE problem
prob = ODEProblem(f!, y0, tspan)

# Solve the ODE using the 4th order Runge-Kutta method
@time sol = solve(prob, RK4(), saveat=t_points)

prob2 = ODEProblem(f!, y0, tspan)

@time sol2 = solve(prob, RK4(), saveat=t_points)

# Plot the solution
plot_1 = plot(sol, xlabel="Time (t)", ylabel="y(t)", title="Solution of ODE using 4th order Runge-Kutta")
println("uff")
savefig(plot_1, "temp.png")


# features and labels generation