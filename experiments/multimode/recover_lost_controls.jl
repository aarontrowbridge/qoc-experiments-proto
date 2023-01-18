using Pico

# loop over files in folder and recover lost controls

data_dir = "data/multimode/free_time/no_guess/problems"
controls_dir = "data/multimode/free_time/no_guess/controls"

for file in readdir(data_dir)

    # remove file extension
    experiment = split(file, ".")[1:end-1] |> s -> join(s, ".")
    println(experiment)

    controls_save_path = joinpath(controls_dir, experiment * ".h5")

    data = load_data(joinpath(data_dir, file))

    save_controls(data, controls_save_path)
end
