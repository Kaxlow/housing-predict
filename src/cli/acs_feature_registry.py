"""Canonical ACS profile features resolved from semantic variable definitions."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ACSFeature:
    name: str
    table: str
    unit: str
    label_pattern: str
    first_year: int = 2010
    last_year: int = 2024
    inflation_adjusted: bool = False
    complement_of_percent: bool = False


def _p(section: str, leaf: str, *, prefix: str = "Percent") -> str:
    # Pre-2013 profile labels commonly omit the section's denominator node even
    # though the variable retains the same universe and meaning.
    if "!!" in leaf:
        denominator, remainder = leaf.split("!!", 1)
        leaf = rf"(?:{denominator}!!)?{remainder}"
    return rf"(?i)^{prefix}(?: Estimate)?!!{section}!!{leaf}$"


ACS_FEATURES = (
    # DP03: economy and employment.
    ACSFeature("labor_force_pct", "dp03", "percent", _p("EMPLOYMENT STATUS", r"Population 16 years and over!!In labor force")),
    ACSFeature("employed_pct", "dp03", "percent", _p("EMPLOYMENT STATUS", r"Population 16 years and over!!In labor force!!Civilian labor force!!Employed")),
    ACSFeature("unemployment_rate_pct", "dp03", "percent", _p("EMPLOYMENT STATUS", r"Civilian labor force!!Unemployment [Rr]ate"), first_year=2015),
    ACSFeature("drove_alone_to_work_pct", "dp03", "percent", _p("COMMUTING TO WORK", r"Workers 16 years and over!!Car, truck, or van -- drove alone")),
    ACSFeature("public_transit_to_work_pct", "dp03", "percent", _p("COMMUTING TO WORK", r"Workers 16 years and over!!Public transportation(?: \(excluding taxicab\))?")),
    ACSFeature("walked_to_work_pct", "dp03", "percent", _p("COMMUTING TO WORK", r"Workers 16 years and over!!Walked")),
    ACSFeature("worked_from_home_pct", "dp03", "percent", _p("COMMUTING TO WORK", r"Workers 16 years and over!!Worked (?:at|from) home")),
    ACSFeature("mean_commute_time_minutes", "dp03", "minutes", _p("COMMUTING TO WORK", r"Workers 16 years and over!!Mean travel time to work \(minutes\)", prefix="Estimate")),
    ACSFeature("management_business_science_arts_occupation_pct", "dp03", "percent", _p("OCCUPATION", r"Civilian employed population 16 years and over!!Management, business, science, and arts occupations")),
    ACSFeature("service_occupation_pct", "dp03", "percent", _p("OCCUPATION", r"Civilian employed population 16 years and over!!Service occupations")),
    ACSFeature("sales_office_occupation_pct", "dp03", "percent", _p("OCCUPATION", r"Civilian employed population 16 years and over!!Sales and office occupations")),
    ACSFeature("natural_resources_construction_maintenance_occupation_pct", "dp03", "percent", _p("OCCUPATION", r"Civilian employed population 16 years and over!!Natural resources, construction, and maintenance occupations")),
    ACSFeature("production_transportation_material_moving_occupation_pct", "dp03", "percent", _p("OCCUPATION", r"Civilian employed population 16 years and over!!Production, transportation, and material moving occupations")),
    ACSFeature("agriculture_mining_industry_pct", "dp03", "percent", _p("INDUSTRY", r"Civilian employed population 16 years and over!!Agriculture, forestry, fishing and hunting, and mining")),
    ACSFeature("construction_industry_pct", "dp03", "percent", _p("INDUSTRY", r"Civilian employed population 16 years and over!!Construction")),
    ACSFeature("manufacturing_industry_pct", "dp03", "percent", _p("INDUSTRY", r"Civilian employed population 16 years and over!!Manufacturing")),
    ACSFeature("finance_real_estate_industry_pct", "dp03", "percent", _p("INDUSTRY", r"Civilian employed population 16 years and over!!Finance and insurance, and real estate and rental and leasing")),
    ACSFeature("professional_services_industry_pct", "dp03", "percent", _p("INDUSTRY", r"Civilian employed population 16 years and over!!Professional, scientific, and management, and administrative and waste management services")),
    ACSFeature("education_health_social_services_industry_pct", "dp03", "percent", _p("INDUSTRY", r"Civilian employed population 16 years and over!!Educational services, and health care and social assistance")),
    ACSFeature("arts_hospitality_food_industry_pct", "dp03", "percent", _p("INDUSTRY", r"Civilian employed population 16 years and over!!Arts, entertainment, and recreation, and accommodation and food services")),
    ACSFeature("median_household_income_2024_usd", "dp03", "2024 USD", r"(?i)^Estimate!!INCOME AND BENEFITS \(IN \d{4} INFLATION-ADJUSTED DOLLARS\)!!(?:Total households!!)?Median household income \(dollars\)$", inflation_adjusted=True),
    ACSFeature("mean_household_earnings_2024_usd", "dp03", "2024 USD", r"(?i)^Estimate!!INCOME AND BENEFITS \(IN \d{4} INFLATION-ADJUSTED DOLLARS\)!!(?:Total households!!)?With earnings!!Mean earnings \(dollars\)$", inflation_adjusted=True),
    ACSFeature("mean_social_security_income_2024_usd", "dp03", "2024 USD", r"(?i)^Estimate!!INCOME AND BENEFITS \(IN \d{4} INFLATION-ADJUSTED DOLLARS\)!!(?:Total households!!)?With Social Security(?: income)?!!Mean Social Security income \(dollars\)$", inflation_adjusted=True),
    ACSFeature("mean_retirement_income_2024_usd", "dp03", "2024 USD", r"(?i)^Estimate!!INCOME AND BENEFITS \(IN \d{4} INFLATION-ADJUSTED DOLLARS\)!!(?:Total households!!)?With retirement income!!Mean retirement income \(dollars\)$", inflation_adjusted=True),
    ACSFeature("mean_supplemental_security_income_2024_usd", "dp03", "2024 USD", r"(?i)^Estimate!!INCOME AND BENEFITS \(IN \d{4} INFLATION-ADJUSTED DOLLARS\)!!(?:Total households!!)?With Supplemental Security Income!!Mean Supplemental Security Income \(dollars\)$", inflation_adjusted=True),
    ACSFeature("mean_cash_public_assistance_income_2024_usd", "dp03", "2024 USD", r"(?i)^Estimate!!INCOME AND BENEFITS \(IN \d{4} INFLATION-ADJUSTED DOLLARS\)!!(?:Total households!!)?With cash public assistance income!!Mean cash public assistance income \(dollars\)$", inflation_adjusted=True),
    ACSFeature("with_health_insurance_pct", "dp03", "percent", _p("HEALTH INSURANCE COVERAGE", r"Civilian noninstitutionalized population!!With health insurance coverage")),
    ACSFeature("without_health_insurance_pct", "dp03", "percent", _p("HEALTH INSURANCE COVERAGE", r"Civilian noninstitutionalized population!!No health insurance coverage")),

    # DP02: household and social characteristics.
    ACSFeature("married_couple_household_pct", "dp02", "percent", _p("HOUSEHOLDS BY TYPE", r"Total households!!Married-couple (?:family|household)"), first_year=2019),
    ACSFeature("cohabiting_couple_household_pct", "dp02", "percent", _p("HOUSEHOLDS BY TYPE", r"Total households!!Cohabiting couple household"), first_year=2020),
    ACSFeature("single_male_householder_pct", "dp02", "percent", _p("HOUSEHOLDS BY TYPE", r"Total households!!Male householder, no (?:wife|spouse/partner) present"), first_year=2019),
    ACSFeature("single_female_householder_pct", "dp02", "percent", _p("HOUSEHOLDS BY TYPE", r"Total households!!Female householder, no (?:husband|spouse/partner) present"), first_year=2019),
    ACSFeature("households_with_children_pct", "dp02", "percent", _p("HOUSEHOLDS BY TYPE", r"Total households!!Households with one or more people under 18 years")),
    ACSFeature("households_with_elderly_pct", "dp02", "percent", _p("HOUSEHOLDS BY TYPE", r"Total households!!Households with one or more people 65 years and over")),
    ACSFeature("average_household_size", "dp02", "people", _p("HOUSEHOLDS BY TYPE", r"Total households!!Average household size", prefix="Estimate")),
    ACSFeature("average_family_size", "dp02", "people", _p("HOUSEHOLDS BY TYPE", r"Total households!!Average family size", prefix="Estimate")),
    ACSFeature("women_with_birth_past_12_months_count", "dp02", "people", r"(?i)^Estimate!!FERTILITY!!Number of women 15 to 50 years old who had a birth in the past 12 months$"),
    ACSFeature("high_school_graduate_or_higher_pct", "dp02", "percent", _p("EDUCATIONAL ATTAINMENT", r"Population 25 years and over!!(?:Percent )?high school graduate or higher")),
    ACSFeature("bachelors_degree_or_higher_pct", "dp02", "percent", _p("EDUCATIONAL ATTAINMENT", r"Population 25 years and over!!(?:Percent )?bachelor's degree or higher")),
    ACSFeature("civilian_veterans_pct", "dp02", "percent", _p("VETERAN STATUS", r"Civilian population 18 years and over!!Civilian veterans")),
    ACSFeature("with_disability_pct", "dp02", "percent", _p("DISABILITY STATUS OF THE CIVILIAN NONINSTITUTIONALIZED POPULATION", r"Total Civilian Noninstitutionalized Population!!With a disability")),
    ACSFeature("moved_different_county_pct", "dp02", "percent", _p("RESIDENCE 1 YEAR AGO", r"Population 1 year and over!!(?:Different house (?:\(in the U\.S\.\)|in the U\.S\.)|Different house \(in the U\.S\. or abroad\)!!Different house in the U\.S\.)!!Different county")),
    ACSFeature("moved_different_state_pct", "dp02", "percent", _p("RESIDENCE 1 YEAR AGO", r"Population 1 year and over!!(?:Different house (?:\(in the U\.S\.\)|in the U\.S\.)|Different house \(in the U\.S\. or abroad\)!!Different house in the U\.S\.)!!Different county!!Different state")),
    ACSFeature("moved_from_abroad_pct", "dp02", "percent", _p("RESIDENCE 1 YEAR AGO", r"Population 1 year and over!!(?:Different house \(in the U\.S\. or abroad\)!!)?Abroad")),
    ACSFeature("limited_english_proficiency_pct", "dp02", "percent", _p("LANGUAGE SPOKEN AT HOME", r"Population 5 years and over!!Language other than English!!Speak English less than \"very well\"")),
    ACSFeature("households_without_computer_pct", "dp02", "percent", _p("COMPUTERS AND INTERNET USE", r"Total households!!With a computer"), first_year=2013, complement_of_percent=True),
    ACSFeature("households_without_broadband_pct", "dp02", "percent", _p("COMPUTERS AND INTERNET USE", r"Total households!!With a broadband Internet subscription"), first_year=2013, complement_of_percent=True),

    # DP04: housing supply, tenure, value, and affordability.
    ACSFeature("housing_units", "dp04", "housing units", _p("HOUSING OCCUPANCY", r"Total housing units", prefix="Estimate")),
    ACSFeature("vacant_housing_units_pct", "dp04", "percent", _p("HOUSING OCCUPANCY", r"Total housing units!!Vacant housing units")),
    ACSFeature("homeowner_vacancy_rate_pct", "dp04", "percent", _p("HOUSING OCCUPANCY", r"Total housing units!!Homeowner vacancy rate", prefix="Estimate")),
    ACSFeature("rental_vacancy_rate_pct", "dp04", "percent", _p("HOUSING OCCUPANCY", r"Total housing units!!Rental vacancy rate", prefix="Estimate")),
    ACSFeature("detached_single_unit_pct", "dp04", "percent", _p("UNITS IN STRUCTURE", r"Total housing units!!1-unit, detached")),
    ACSFeature("large_multifamily_20_plus_units_pct", "dp04", "percent", _p("UNITS IN STRUCTURE", r"Total housing units!!20 or more units")),
    ACSFeature("mobile_home_pct", "dp04", "percent", _p("UNITS IN STRUCTURE", r"Total housing units!!Mobile home")),
    ACSFeature("built_1939_or_earlier_pct", "dp04", "percent", _p("YEAR STRUCTURE BUILT", r"Total housing units!!Built 1939 or earlier")),
    ACSFeature("median_rooms_per_unit", "dp04", "rooms", _p("ROOMS", r"Total housing units!!Median rooms", prefix="Estimate")),
    ACSFeature("owner_occupied_pct", "dp04", "percent", _p("HOUSING TENURE", r"Occupied housing units!!Owner-occupied")),
    ACSFeature("renter_occupied_pct", "dp04", "percent", _p("HOUSING TENURE", r"Occupied housing units!!Renter-occupied")),
    ACSFeature("average_owner_household_size", "dp04", "people", _p("HOUSING TENURE", r"Occupied housing units!!Average household size of owner-occupied unit", prefix="Estimate")),
    ACSFeature("average_renter_household_size", "dp04", "people", _p("HOUSING TENURE", r"Occupied housing units!!Average household size of renter-occupied unit", prefix="Estimate")),
    ACSFeature("householder_moved_1989_or_earlier_pct", "dp04", "percent", _p("YEAR HOUSEHOLDER MOVED INTO UNIT", r"Occupied housing units!!Moved in 1989 (?:or|and) earlier"), first_year=2019),
    ACSFeature("target_median_owner_occupied_home_value_2024_usd", "dp04", "2024 USD", r"(?i)^Estimate!!VALUE!!(?:Owner-occupied units!!)?Median \(dollars\)$", inflation_adjusted=True),
    ACSFeature("with_mortgage_pct", "dp04", "percent", _p("MORTGAGE STATUS", r"Owner-occupied units!!Housing units with a mortgage")),
    ACSFeature("without_mortgage_pct", "dp04", "percent", _p("MORTGAGE STATUS", r"Owner-occupied units!!Housing units without a mortgage")),
    ACSFeature("owner_cost_30_to_34_9_pct", "dp04", "percent", _p(r"SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME \(SMOCAPI\)", r"Housing units with a mortgage \(excluding units where SMOCAPI cannot be computed\)!!30\.0 to 34\.9 percent")),
    ACSFeature("owner_cost_35_plus_pct", "dp04", "percent", _p(r"SELECTED MONTHLY OWNER COSTS AS A PERCENTAGE OF HOUSEHOLD INCOME \(SMOCAPI\)", r"Housing units with a mortgage \(excluding units where SMOCAPI cannot be computed\)!!35\.0 percent or more")),
    ACSFeature("gross_rent_30_to_34_9_pct", "dp04", "percent", _p(r"GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME \(GRAPI\)", r"Occupied units paying rent \(excluding units where GRAPI cannot be computed\)!!30\.0 to 34\.9 percent")),
    ACSFeature("gross_rent_35_plus_pct", "dp04", "percent", _p(r"GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME \(GRAPI\)", r"Occupied units paying rent \(excluding units where GRAPI cannot be computed\)!!35\.0 percent or more")),
)
