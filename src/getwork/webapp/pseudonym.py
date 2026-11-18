"""Placeholder candidate names for the demo (ADR-40, CC-21).

The resume corpus has no names, so the UI shows a name derived from the resume id. It is
computed at display time, never stored, embedded, ranked on or sent to the chat agent.
"""

from __future__ import annotations

import hashlib

NAME_SEED: str = "getwork-demo-names-v1"

FIRST_NAMES: tuple[str, ...] = (
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Daniel", "Karen", "Matthew", "Nancy", "Anthony", "Lisa",
    "Mark", "Betty", "Donald", "Margaret", "Steven", "Sandra", "Paul", "Ashley",
    "Andrew", "Kimberly", "Joshua", "Emily", "Kenneth", "Donna", "Kevin", "Michelle",
    "Brian", "Carol", "Edward", "Amanda", "George", "Melissa", "Timothy", "Deborah",
    "Ronald", "Stephanie", "Jason", "Rebecca", "Jeffrey", "Sharon", "Ryan", "Laura",
    "Jacob", "Cynthia", "Gary", "Kathleen", "Nicholas", "Amy", "Eric", "Angela",
    "Jonathan", "Shirley", "Stephen", "Anna", "Larry", "Brenda", "Justin", "Pamela",
    "Scott", "Emma", "Brandon", "Nicole", "Benjamin", "Helen", "Samuel", "Samantha",
    "Gregory", "Katherine", "Alexander", "Christine", "Frank", "Debra", "Patrick", "Rachel",
    "Raymond", "Carolyn", "Jack", "Janet", "Dennis", "Maria", "Jerry", "Olivia",
    "Tyler", "Heather", "Aaron", "Diane", "Jose", "Julie", "Adam", "Joyce",
    "Nathan", "Victoria", "Henry", "Kelly", "Douglas", "Christina", "Zachary", "Lauren",
    "Peter", "Joan", "Kyle", "Evelyn", "Walter", "Judith", "Ethan", "Andrea",
    "Jeremy", "Hannah", "Harold", "Megan", "Keith", "Cheryl", "Christian", "Jacqueline",
    "Roger", "Martha", "Noah", "Gloria", "Gerald", "Teresa", "Carl", "Sara",
    "Terry", "Janice", "Sean", "Marie", "Austin", "Julia", "Arthur", "Grace",
    "Lawrence", "Judy", "Jesse", "Theresa", "Dylan", "Madison", "Bryan", "Beverly",
    "Joe", "Denise", "Jordan", "Marilyn", "Billy", "Amber", "Bruce", "Danielle",
    "Albert", "Rose", "Willie", "Brittany", "Gabriel", "Diana", "Logan", "Abigail",
    "Alan", "Natalie", "Juan", "Jane", "Priya", "Wei", "Mei", "Arjun",
    "Aisha", "Omar", "Fatima", "Yusuf", "Sofia", "Mateo", "Lucia", "Ravi",
    "Anjali", "Hiroshi", "Yuki", "Chen", "Ling", "Ahmed", "Layla", "Diego",
    "Carmen", "Ivan", "Elena", "Kwame", "Amara",
)

LAST_NAMES: tuple[str, ...] = (
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
    "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
    "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
    "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
    "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy",
    "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey",
    "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson",
    "Watson", "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza",
    "Ruiz", "Hughes", "Price", "Alvarez", "Castillo", "Sanders", "Patel", "Myers",
    "Long", "Ross", "Foster", "Jimenez", "Powell", "Jenkins", "Perry", "Russell",
    "Sullivan", "Bell", "Coleman", "Butler", "Henderson", "Barnes", "Gonzales", "Fisher",
    "Vasquez", "Simmons", "Romero", "Jordan", "Patterson", "Alexander", "Hamilton", "Graham",
    "Reynolds", "Griffin", "Wallace", "Moreno", "West", "Cole", "Hayes", "Bryant",
    "Herrera", "Gibson", "Ellis", "Tran", "Medina", "Aguilar", "Stevens", "Murray",
    "Ford", "Castro", "Marshall", "Owens", "Harrison", "Fernandez", "Donovan", "Woods",
    "Washington", "Kennedy", "Wells", "Vargas", "Henry", "Chen", "Freeman", "Webb",
    "Tucker", "Guzman", "Burns", "Crawford", "Olson", "Simpson", "Porter", "Hunter",
    "Gordon", "Mendez", "Silva", "Shaw", "Snyder", "Mason", "Dixon", "Munoz",
    "Hunt", "Hicks", "Holmes", "Palmer", "Wagner", "Black", "Robertson", "Boyd",
    "Rose", "Stone", "Salazar", "Fox", "Warren", "Mills", "Meyer", "Rice",
    "Schmidt", "Garza", "Daniels", "Ferguson", "Nichols", "Stephens", "Soto", "Weaver",
    "Ryan", "Gardner", "Payne", "Grant", "Dunn", "Kelley", "Spencer", "Hawkins",
    "Nair", "Singh", "Khan", "Ali", "Wang", "Zhang", "Liu", "Yamamoto",
    "Tanaka", "Okafor", "Mensah",
)


def display_name(doc_id: str | None) -> str | None:
    """Placeholder "First Last" for a document id; None when the id is None or blank.

    The id is hashed exactly as given (no strip, no case fold) with SHA-256, not hash(),
    so the name is the same in every process.
    """
    if doc_id is None or doc_id.strip() == "":
        return None
    digest = hashlib.sha256(f"{NAME_SEED}:{doc_id}".encode("utf-8")).digest()
    first = FIRST_NAMES[int.from_bytes(digest[0:4], "big") % len(FIRST_NAMES)]
    last = LAST_NAMES[int.from_bytes(digest[4:8], "big") % len(LAST_NAMES)]
    return f"{first} {last}"
