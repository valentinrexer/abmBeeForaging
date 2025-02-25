'''

def anticipation(self, method, time_source_found, days_of_experience, sunrise, sunset, distance):
    """
    Simulating the bees anticipation behaviour based on her experience

    :param method: Anticipation method
    :param time_source_found: time that the bee found the source
    :param days_of_experience: Days passed since the bee found the source
    :param sunrise: sunrise time for this simulation
    :param sunset: sunset time for this simulation
    :return: anticipated time for soucre
    """

    if not sunrise <= time_source_found <= sunset:
        warnings.warn("Invalid argument for time_source_found")

    if method == 1:
        return time_source_found - distance / FLYING_SPEED

    elif method == 2:

        return (time_source_found - 3600 * 4 * (time_source_found - sunrise) / (
                    sunset - sunrise)) - distance / FLYING_SPEED

    elif method == 3 or method == 4:

        subtract_anticipation = [3600 * 2, 3600 * 1.5, 3600]
        initial_anticipation = time_source_found - subtract_anticipation[days_of_experience - 1]

        if method == 4:
            initial_anticipation -= subtract_anticipation[days_of_experience - 1]

        if initial_anticipation < sunrise:
            return sunrise

        else:
            return initial_anticipation - distance / FLYING_SPEED

    elif method == 5:
        subtract_factor = [4, 3, 2]
        return time_source_found - 3600 * subtract_factor[days_of_experience - 1] * (
                time_source_found - sunrise) / (sunset - sunrise)


    else:
        return -1


'''