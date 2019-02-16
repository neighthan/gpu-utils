# TODO - describe the format a bit more here

# available attributes: idx, util_used, util_free, mem_used, mem_free
base_format = ["[{idx}]", "{util_used: >3} %", "{mem_free: >5}"]
sep = " | "  # separator between the attributes in the list above
# either a color for each attribute or a function which takes in the attribute
# and returns a color. Use "" to not add a color.
colors = ["{fg('light_cyan')}", "{fg('green')}", "{fg('yellow_1')}"]

# either replace <max_cmd_width> directly below or pass in a value from the command line
# if set here, it can't be overridden from the command line. The default value is 125

# available attributes: user, gpu_mem_used, command
process_base_format = ["{user}", "{gpu_mem_used}", "{command:.<max_cmd_width>}"]
process_sep = " "
process_colors = ["{fg('magenta_3a')}", "{fg('yellow')}", ""]
