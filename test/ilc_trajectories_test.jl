using Pico
using Test

# @testset
Z = rand(5, 10)

components = (
    x = 1:2,
    y = 3:5,
)

traj = Traj(Z, components)

zs = traj[2:4]

zs[2].z

traj.components

add_component!(traj, :z, rand(3, 10))

traj

traj[2].z
