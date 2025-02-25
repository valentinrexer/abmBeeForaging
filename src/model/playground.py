import mesa
import grid
import warnings


def anticipation(method, time_source_found, days_of_experience, sunrise, sunset):
    if not sunrise <= time_source_found <= sunset:
        warnings.warn("Invalid argument for time_source_found")

    if method == 1:
        return time_source_found

    elif method == 2:

        return time_source_found - 3600 * 4 * (time_source_found - sunrise) / (sunset - sunrise)

    elif method == 3 or method == 4:

        subtract_anticipation = [3600 * 2, 3600 * 1.5, 3600]
        initial_anticipation = time_source_found - subtract_anticipation[days_of_experience-1]

        if method == 4:
            initial_anticipation -= subtract_anticipation[days_of_experience-1]

        if initial_anticipation < sunrise:
            return sunrise

        else:
            return initial_anticipation

    elif method >= 5:
        subtract_factor = [4, 3, 2]
        return time_source_found - 3600 * subtract_factor[days_of_experience-1] * (time_source_found - sunrise) / (sunset - sunrise)


    else:
        return -1




def to_daytime(time):
    return time / 3600

def to_ticks(time):
    return time * 3600



print(to_daytime(anticipation(5, to_ticks(19),1, to_ticks(7), to_ticks(19))))







