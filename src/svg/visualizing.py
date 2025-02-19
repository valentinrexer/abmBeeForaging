'''model_instance = BeeForagingModel(900)
    svg_string = ""
    bee = None


    for _ in range(1):
        forager = ForagerBeeAgent(_, model_instance, BeeForagerType.PERSISTENT, 1, model_instance.grid.hive)
        forager.status = BeeStatus.FLYING_TO_SEARCH_AREA
        forager.targeted_position = (model_instance.flower_location[0] + random.randint(-35,35), model_instance.flower_location[1] + random.randint(-35,35))
        model_instance.grid.place_agent(forager, model_instance.grid.hive)
        model_instance.schedule.add(forager)
        bee = forager

    svg_string += "M" + str(model_instance.grid.hive[0])+","+str(model_instance.grid.hive[1])+" "
    for step in range(960):
        model_instance.schedule.step()
        for forager in model_instance.schedule.agents:
            if isinstance(forager, ForagerBeeAgent) and forager.unique_id == 0:
                x = forager.accurate_position[0]
                y = forager.accurate_position[1]
                x = int(x*100) / 100.0
                y = int(y*100) / 100.0


                svg_string += "L"+str(x)+","+str(y)+" "



    file_string = ""
    with open("/home/valentin-rexer/uni/UofM/abm_files/sample.svg", 'r') as sample:
        for line in sample:
            file_string += line

    file_string += svg_string + '"'



    file_string += """
         fill="none"
          stroke="#2563eb"
          stroke-width="0.5"
          stroke-linecap="round"
          stroke-linejoin="round" />
    """

    file_string += "\n<!-- Dots -->\n"

    # Add circles with variables
    file_string += f'<circle cx="{model_instance.flower_location[0]}" cy="{model_instance.flower_location[1]}" r="{1}" fill="#4CAF50" /> <!-- Circle -->\n'
    file_string += f'\n<circle cx="{forager.targeted_position[0]}" cy="{forager.targeted_position[1]}" r="{MAX_SEARCH_RADIUS}" fill="none" stroke="#4CAF50" stroke-width="{0.3}" />\n'

    file_string += "\n</svg>"

    with open("/home/valentin-rexer/uni/UofM/abm_files/final.svg", "w") as final:
        final.write(file_string)

    print("flower")
    print(model_instance.flower_location)

    if bee.accurate_position is model_instance.flower_location:
        print("ounf")

'''