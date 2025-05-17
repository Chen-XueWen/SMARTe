import os
import ast
import re
from tqdm import tqdm
import argparse
import pickle
import torch
from openai import OpenAI
from keys import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI()

def extract_triples_with_llm(text):
    """
    Prompts the model to extract triples in JSON-like list-of-dicts format,
    choosing relations only from the given 218-item inventory.
    Returns: a Python list of dicts (each with 'subject','relation','object').
    """
    RELATIONS = [
        "region", "cityServed", "ISBN_number", "chairman", "class", "administrativeArrondissement", "bird", "influencedBy", "notableWork", "was given the 'Technical Campus' status by", "ICAO_Location_Identifier", "hubAirport", "leader", "affiliation", "areaOfLand", "firstPublicationYear", "inaugurationDate", "5th_runway_Number", "director", "leaderParty", "location", "3rd_runway_LengthFeet", "populationDensity", "campus", "firstAppearanceInFilm", "state", "areaOfWater", "rector", "numberOfMembers", "religion", "elevationAboveTheSeaLevel_(in_feet)", "headquarter", "mediaType", "EISSN_number", "title", "placeOfBirth", "creatorOfDish", "parentCompany", "served as Chief of the Astronaut Office in", "has to its southwest", "currentTenants", "series", "address", "LibraryofCongressClassification", "runwayLength", "fossil", "languages", "1st_runway_SurfaceType", "postalCode", "child", "foundationPlace", "course", "has to its southeast", "areaTotal", "season", "occupation", "nativeName", "material", "3rd_runway_SurfaceType", "abbreviation", "youthclub", "ReferenceNumber in the National Register of Historic Places", "language", "yearOfConstruction", "servingTemperature", "city", "birthPlace", "genus", "was selected by NASA", "populationTotal", "president", "distributor", "part", "fullname", "backup pilot", "aircraftFighter", "OCLC_number", "voice", "significantProject", "nearestCity", "CODEN_code", "numberOfRooms", "hometown", "chief", "senators", "protein", "birthName", "nationality", "has to its northwest", "ISSN_number", "neighboringMunicipality", "almaMater", "legislature", "country", "headquarters", "governmentType", "author", "municipality", "author", "precededBy", "transportAircraft", "fullName", "representative", "residence", "doctoralAdvisor", "significantBuilding", "transportAircraft", "district", "dedicatedTo", "chairmanTitle", "anthem", "operator", "isPartOf", "4th_runway_LengthFeet", "mainIngredients", "areaCode", "academicDiscipline", "architect", "gemstone", "year", "commander", "LCCN_number", "deathPlace", "chairperson", "buildingStartDate", "order", "patronSaint", "officialLanguage", "runwayName", "1st_runway_LengthMetre", "owningOrganisation", "followedBy", "floorCount", "currency", "editor", "jurisdiction", "founder", "crewMembers", "champions", "elevationAboveTheSeaLevel_(in_metres)", "largestCity", "numberOfUndergraduateStudents", "chancellor", "1st_runway_LengthFeet", "height", "administrativeCounty", "locationCity", "mayor", "category", "tenant", "numberOfPages", "starring", "publisher", "established", "keyPerson", "fat", "governingBody", "elevationAboveTheSeaLevel", "manager", "nickname", "sportsGoverningBody", "capital", "1st_runway_Number", "battles", "family", "numberOfPostgraduateStudents", "placeOfDeath", "genre", "foundedBy", "award", "demonym", "affiliations", "ethnicGroup", "regionServed", "ground", "architecturalStyle", "was a crew member of", "alternativeName", "latinName", "dean", "creator", "bedCount", "leaderName", "aircraftHelicopter", "ethnicGroups", "compete in", "IATA_Location_Identifier", "developer", "academicStaffSize", "2nd_runway_SurfaceType", "dishVariation", "spokenIn", "awards", "sportsOffered", "floorArea", "architecture", "has to its north", "product", "designer", "has to its west", "leaderTitle", "numberOfStudents", "river", "countySeat", "owner", "has to its northeast", "operatingOrganisation", "motto", "higher", "motto", "club", "added to the National Register of Historic Places", "outlookRanking", "broadcastedBy", "completionDate", "ingredient", "attackAircraft", "league"
    ]
    rels_str = ", ".join(f"'{r}'" for r in RELATIONS)

    prompt = f"""You are a relation-triple extractor.  Given the following sentence, extract all relational triples.
Each triple must be a dict with exactly three keys: 'subject', 'relation', 'object'.
The 'relation' value must be ONE of the following 24 options:
{rels_str}
Sentence:\"\"\"{text}\"\"\"
Output ONLY a Python list of dictionaries.  For example
[{{'subject':'Bobby Fischer','relation':'nationality','object':'Iceland'}}
 {{'subject':'Iceland','relation':'capital','object':'Reykjavik'}}]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
          {
            "role": "user",
            "content": f"{prompt}"
          }
        ]
    )
    
    raw = response.choices[0].message.content
    raw = re.sub(r"^```python\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
    print(text)
    print(raw) # For troubleshooting

    # Try to parse as a Python literal
    try:
        triples = ast.literal_eval(raw.strip())
        if isinstance(triples, list):
            return triples
    except Exception:
        pass

    # Fallback: regex-based crude extraction of lines like subject|relation|object
    triples = []
    for line in raw.splitlines():
        parts = re.split(r"\s*\|\s*", line.strip())
        if len(parts) == 3:
            subj, rel, obj = parts
            triples.append({'subject': subj, 'relation': rel, 'object': obj})
    return triples

def compute_f1(true_triples, pred_triples):
    """
    Compute precision, recall, F1 between two lists of dicts.
    Treat each dict as a tuple (subject, relation, object).
    """
    true_set = set((t['subject'],t['relation'],t['object']) for t in true_triples)
    pred_set = set((t['subject'],t['relation'],t['object']) for t in pred_triples)
    tp = len(true_set & pred_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(true_set) if true_set else 0.0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
    return p, r, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="WebNLG-Exact", help="Dataset type (e.g., NYT-Exact)")
    args = parser.parse_args()

    #with open(f"./processed/{args.data_type}/traindev_features.pkl", 'rb') as f:
    #    re_training = pickle.load(f)
    with open(f"./processed/{args.data_type}/test_features.pkl", 'rb') as f:
        re_testing = pickle.load(f)

    all_precisions, all_recalls, all_f1s = [], [], []

    for sample in tqdm(re_testing, total=len(re_testing), desc="Evaluating"):
        text = sample[1]
        gold = sample[2]
        pred = extract_triples_with_llm(text)

        p, r, f1 = compute_f1(gold, pred)
        all_precisions.append(p)
        all_recalls.append(r)
        all_f1s.append(f1)

    # report
    avg_p = sum(all_precisions) / len(all_precisions)
    avg_r = sum(all_recalls)    / len(all_recalls)
    avg_f1 = sum(all_f1s)       / len(all_f1s)
    print(f"Zero-Shot on {len(re_testing)} samples →  P: {avg_p:.3f}  R: {avg_r:.3f}  F1: {avg_f1:.3f}")

if __name__ == "__main__":
    main()
