import mesa
import grid


def anticipation(method, , time_source_found, days_of_experience, sunrise, sunset):
    if method == 1:
        return time_source_found

    elif method == 2:
        return time_source_found - 4 * (time_source_found - sunrise) / (sunset - sunrise)

    elif method == 3 or 4:
        subtract_anticipation = [3600 * 2, 3600 * 1.5, 3600]
        initial_anticipation = time_source_found - subtract_anticipation[days_of_experience]

        if method == 4:
            initial_anticipation -= subtract_anticipation[days_of_experience]

        if initial_anticipation < sunrise:
            return sunrise

        else:
            return initial_anticipation


    elif method == 5:
        subtract_factor = [4, 3, 2]
        return time_source_found - ((time_source_found - sunrise) * (subtract_factor[days_of_experience] / (sunset-sunrise)/3600))



def clustering_time()


