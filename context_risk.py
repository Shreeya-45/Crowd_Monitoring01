# context_risk.py
# Context aware crowd risk analysis
# Different places have different crowd tolerance


PLACE_LIMITS = {

    "School": {
        "low": 80,
        "medium": 150,
        "high": 250
    },


    "Railway Station": {
        "low": 150,
        "medium": 400,
        "high": 800
    },


    "Mall": {
        "low": 200,
        "medium": 500,
        "high": 1000
    },


    "Hospital": {
        "low": 50,
        "medium": 120,
        "high": 250
    },


    "Stadium": {
        "low": 500,
        "medium": 2000,
        "high": 5000
    },


    "Office": {
        "low": 15,
        "medium": 30,
        "high": 50
    }

}



# Default selected location
CURRENT_PLACE = "School"



# -----------------------------
# Set selected place from UI
# -----------------------------

def set_place(place):

    global CURRENT_PLACE


    if place in PLACE_LIMITS:

        CURRENT_PLACE = place



# -----------------------------
# Get current place
# -----------------------------

def get_place():

    return CURRENT_PLACE



# -----------------------------
# Get limits of current place
# -----------------------------

def get_limits():

    return PLACE_LIMITS[CURRENT_PLACE]



# -----------------------------
# Main risk calculation
# -----------------------------

def get_context_risk(person_count):


    limits = get_limits()



    if person_count < limits["low"]:

        return "LOW"



    elif person_count < limits["medium"]:

        return "MODERATE"



    elif person_count < limits["high"]:

        return "HIGH"



    else:

        return "CRITICAL"




# -----------------------------
# Used for density trend
# Converts crowd count into %
# based on place capacity
# -----------------------------

def get_place_capacity():

    return PLACE_LIMITS[CURRENT_PLACE]["medium"]



def get_normalized_density(person_count):

    capacity = get_place_capacity()


    if capacity == 0:

        return 0



    percentage = (
        person_count / capacity
    ) * 100



    return min(
        percentage,
        150
    )