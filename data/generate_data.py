"""
Movie Recommendation System - Synthetic Data Generator
=======================================================
Generates realistic synthetic datasets for 450 movies, 200 users, and 4000+ ratings.
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# ─── Constants ─────────────────────────────────────────────────────────────────

ALL_GENRES = [
    "Action", "Comedy", "Drama", "Sci-Fi", "Horror",
    "Romance", "Thriller", "Animation", "Adventure", "Fantasy",
    "Crime", "Mystery", "Biography", "History", "War",
    "Western", "Musical", "Sport", "Family", "Documentary"
]

DIRECTORS = [
    "Christopher Nolan", "Steven Spielberg", "Quentin Tarantino", "Martin Scorsese",
    "James Cameron", "Ridley Scott", "David Fincher", "Wes Anderson",
    "Denis Villeneuve", "Greta Gerwig", "Jordan Peele", "Ava DuVernay",
    "Alfonso Cuarón", "Park Chan-wook", "Bong Joon-ho", "Guillermo del Toro",
    "Peter Jackson", "Francis Ford Coppola", "Billy Wilder", "Clint Eastwood",
    "Tim Burton", "Ang Lee", "David Lynch", "Ron Howard",
    "Paul Thomas Anderson", "Darren Aronofsky", "Sofia Coppola", "Spike Lee",
    "Oliver Stone", "Sam Mendes", "David O. Russell", "Bennett Miller",
    "Jason Reitman", "Alex Garland", "Ryan Coogler", "Jon Favreau",
    "J.J. Abrams", "Guy Ritchie", "Edgar Wright", "Joe Wright",
    "Danny Boyle", "Christopher McQuarrie", "Gore Verbinski", "Antoine Fuqua",
    "F. Gary Gray", "John Woo", "Tony Scott", "Roland Emmerich",
    "Michael Bay", "Zack Snyder"
]

ACTORS = [
    "Leonardo DiCaprio", "Meryl Streep", "Tom Hanks", "Scarlett Johansson",
    "Denzel Washington", "Cate Blanchett", "Brad Pitt", "Natalie Portman",
    "Morgan Freeman", "Emma Stone", "Robert De Niro", "Jennifer Lawrence",
    "Christian Bale", "Viola Davis", "Matt Damon", "Zendaya",
    "Ryan Gosling", "Ana de Armas", "Oscar Isaac", "Lupita Nyong'o",
    "Timothée Chalamet", "Florence Pugh", "Adam Driver", "Margot Robbie",
    "Tom Cruise", "Keanu Reeves", "Will Smith", "Al Pacino",
    "Johnny Depp", "Chris Pratt", "Ryan Reynolds", "Robert Downey Jr.",
    "Chris Evans", "Samuel L. Jackson", "Idris Elba", "Michael B. Jordan",
    "Mahershala Ali", "Jake Gyllenhaal", "Andrew Garfield", "Anya Taylor-Joy",
    "Pedro Pascal", "John Boyega", "Dwayne Johnson", "Vin Diesel",
    "Jason Statham", "Mark Wahlberg", "Ben Affleck", "Hugh Jackman",
    "Javier Bardem", "Daniel Craig", "Liam Neeson", "Chris Hemsworth",
    "Zoe Saldana", "Gal Gadot", "Brie Larson", "Jeremy Renner"
]

# 450 movie titles spanning many genres and years
MOVIE_TITLES = [
    # Iconic classics
    "Inception", "The Dark Knight", "Interstellar", "Parasite", "The Matrix",
    "Pulp Fiction", "Forrest Gump", "The Shawshank Redemption", "Avengers: Endgame",
    "The Godfather", "Schindler's List", "Goodfellas", "Fight Club",
    "The Silence of the Lambs", "Casablanca", "Jurassic Park", "Titanic",
    "The Lion King", "Toy Story", "WALL-E", "Up", "Finding Nemo",
    "Blade Runner 2049", "Arrival", "Dune", "Mad Max: Fury Road",
    "Get Out", "Us", "Nope", "Everything Everywhere All at Once",
    "The Grand Budapest Hotel", "Moonrise Kingdom", "Fantastic Mr. Fox",
    "Whiplash", "La La Land", "First Man", "Tenet", "Memento",
    "The Prestige", "Dunkirk", "Oppenheimer", "Barbie",
    "Spider-Man: Into the Spider-Verse", "Soul", "Encanto", "Moana",
    "Knives Out", "Glass Onion", "The Revenant", "1917",
    # Action / Thriller
    "John Wick", "John Wick Chapter 2", "John Wick Chapter 3", "John Wick Chapter 4",
    "Mission: Impossible", "Mission: Impossible Fallout", "Mission: Impossible Dead Reckoning",
    "Top Gun: Maverick", "The Batman", "No Time to Die",
    "Black Panther", "Black Panther: Wakanda Forever", "Thor: Ragnarok",
    "Captain America: Civil War", "Avengers: Infinity War",
    "Iron Man", "Doctor Strange", "Guardians of the Galaxy",
    "Spider-Man: Homecoming", "Spider-Man: No Way Home",
    "Wonder Woman", "Aquaman", "The Suicide Squad",
    "Free Guy", "Bullet Train", "The Gray Man",
    "Ambulance", "The Lost City", "Prey",
    "Logan", "Deadpool", "Deadpool & Wolverine",
    # Drama
    "The Father", "Nomadland", "Minari", "Promising Young Woman",
    "Green Book", "Moonlight", "12 Years a Slave", "Birdman",
    "Spotlight", "The Revenant", "Room", "Brooklyn",
    "Marriage Story", "The Irishman", "Once Upon a Time in Hollywood",
    "Joker", "1917", "Ford v Ferrari", "Little Women",
    "Midsommar", "Hereditary", "Us", "mother!",
    "Black Swan", "Requiem for a Dream", "Pi", "The Wrestler",
    "Silver Linings Playbook", "American Hustle", "Three Billboards",
    "The Shape of Water", "Lady Bird", "Phantom Thread",
    "Call Me by Your Name", "Moonlight", "Manchester by the Sea",
    "Spotlight", "The Big Short", "The Martian",
    "Gravity", "Her", "The Wolf of Wall Street",
    "Django Unchained", "Lincoln", "Zero Dark Thirty",
    "Argo", "The Artist", "The Tree of Life",
    # Horror
    "Hereditary", "Midsommar", "The Witch", "Get Out", "Us",
    "Nope", "X", "Scream", "Halloween Ends", "Smile",
    "M3GAN", "The Menu", "Knock at the Cabin", "Barbarian",
    "Pearl", "Talk to Me", "Skinamarink", "The Black Phone",
    "Crimes of the Future", "The Northman",
    "Doctor Sleep", "It", "It Chapter Two",
    "A Quiet Place", "A Quiet Place Part II",
    "Bird Box", "Don't Look Up", "The Invisible Man",
    "Crawl", "Ready or Not", "The Lodge",
    "The Autopsy of Jane Doe", "The Ritual", "Annihilation",
    # Sci-Fi
    "Dune: Part Two", "Avatar: The Way of Water",
    "Everything Everywhere All at Once", "Nope",
    "Moonfall", "The Tomorrow War", "Infinite",
    "Finch", "The Midnight Sky", "Love and Monsters",
    "Army of the Dead", "Zack Snyder's Justice League",
    "Mortal Kombat", "Space Jam: A New Legacy",
    "Snake Eyes", "F9", "Fast X",
    "Venom", "Venom: Let There Be Carnage",
    "Morbius", "Uncharted", "Borderlands",
    "Alien: Romulus", "Kingdom of the Planet of the Apes",
    "Transformers One", "The Wild Robot",
    "Rebel Moon", "Rebel Moon Part 2",
    "Atlas", "Spaceman",
    # Animation
    "Raya and the Last Dragon", "Luca", "Turning Red",
    "Lightyear", "Strange World", "Wish",
    "Puss in Boots: The Last Wish", "The Bad Guys",
    "DC League of Super-Pets", "Sonic the Hedgehog 2",
    "Chip 'n Dale: Rescue Rangers", "The Sea Beast",
    "My Father's Dragon", "Wendell & Wild",
    "Marcel the Shell with Shoes On", "Apollo 10½",
    "Flee", "The Mitchells vs. the Machines",
    "Wolfwalkers", "Onward",
    "Frozen II", "Toy Story 4", "The Incredibles 2",
    "Coco", "Zootopia", "Big Hero 6",
    "Inside Out", "The Good Dinosaur", "Brave",
    "Tangled", "Wreck-It Ralph", "Ralph Breaks the Internet",
    # Comedy
    "The Nice Guys", "Game Night", "Palm Springs",
    "The Disaster Artist", "I, Tonya", "Blockers",
    "Booksmart", "Rocketman", "Yesterday",
    "Long Shot", "Jojo Rabbit", "Portrait of a Lady on Fire",
    "The Peanut Butter Falcon", "Official Secrets",
    "Downton Abbey", "Rocketman", "Blinded by the Light",
    "Late Night", "Late Night with the Devil",
    "Theater Camp", "Bottoms", "Cocaine Bear",
    "The Menu", "Triangle of Sadness", "Banshees of Inisherin",
    "The Fabelmans", "TÁR", "Women Talking",
    "Till", "Causeway", "Emily the Criminal",
    # Crime / Mystery
    "The French Dispatch", "No Sudden Move",
    "Judas and the Black Messiah", "The Harder They Fall",
    "The Inspection", "Passing", "Zola",
    "Flag Day", "Pig", "The Card Counter",
    "Mass", "The Last Duel", "The Power of the Dog",
    "Licorice Pizza", "Drive My Car", "Belfast",
    "CODA", "Dune", "The Tragedy of Macbeth",
    "Being the Ricardos", "tick, tick...BOOM!",
    "The Eyes of Tammy Faye", "King Richard", "Spencer",
    # Recent hits
    "Poor Things", "American Fiction", "The Holdovers", "Maestro", "Nyad",
    "Origin", "Rustin", "Shirley", "The Color Purple",
    "Monkey Man", "Civil War", "Furiosa",
    "Hit Man", "Bad Boys: Ride or Die",
    "Inside Out 2", "Despicable Me 4", "Twisters",
    "Alien: Romulus", "Speak No Evil", "Longlegs",
    "The Substance", "Conclave", "Heretic",
    "Gladiator II", "Wicked", "Moana 2",
    "Red One", "Transformers One",
    "The Apprentice", "The Brutalist",
    "A Complete Unknown", "Better Man", "Nickel Boys",
    "Maria", "Hard Truths", "One of Them Days",
    # Additional variety
    "Shadow Land", "Crimson Rivers", "Broken Chains", "Neon Mirage", "Iron Storm",
    "Silent Echo", "Golden Horizon", "Dark Waters", "Rising Sun", "Fallen Angel",
    "Steel Resolve", "Crystal Palace", "Thunder Road", "Midnight Express", "Fire Storm",
    "Cold Silence", "Burning Bridges", "Lost Highway", "Endless Night", "Frozen Hearts",
    "Blood Diamond", "Rough Justice", "Stone Cold", "Dead Reckoning", "Final Verdict",
    "Sacred Ground", "Hidden Depths", "Shattered Dreams", "Broken Glass", "Perfect Storm",
    "Ocean Deep", "Mountain Peak", "Desert Wind", "Forest Dark", "City of Shadows",
    "River of No Return", "Island of Secrets", "Valley of Kings", "Tower of Babel", "Garden of Eden",
    "Gates of Heaven", "Road to Perdition", "Path of Destruction", "Bridge to Nowhere", "Wall of Silence",
    "Circle of Life", "Chain Reaction", "Point of No Return", "Line of Fire", "Edge of Tomorrow",
]

DESCRIPTIONS = [
    "A mind-bending thriller about dreams within dreams that challenges the nature of reality.",
    "A caped crusader battles chaos in Gotham City against an anarchic villain.",
    "Astronauts traverse a wormhole near Saturn to save a dying humanity.",
    "A dark comedy about class warfare between two families in Korean society.",
    "A hacker discovers the truth behind a simulated reality and joins a rebellion.",
    "Interconnected crime stories in Los Angeles told through a non-linear narrative.",
    "A simple man with extraordinary luck goes through pivotal moments in American history.",
    "Two wrongfully convicted men forge an unlikely friendship in a Maine prison.",
    "Earth's mightiest heroes unite against a cosmic threat to save the universe.",
    "A crime family dynasty rises and falls in mid-20th century America.",
    "One man's mission to save Jewish lives during the Holocaust in Nazi Germany.",
    "The rise and fall of a mob associate in New York's criminal underworld.",
    "A soap salesman forms an underground fight club that grows into something far darker.",
    "An FBI trainee faces a brilliant but terrifying cannibal psychiatrist to catch a serial killer.",
    "A timeless love story between an American and a Bulgarian refugee during World War II.",
    "Scientists and a boy must survive when dinosaurs escape on a tropical island park.",
    "A doomed love story between a rich man and a poor woman aboard an ill-fated ocean liner.",
    "A young lion prince must reclaim his kingdom after his uncle murders his father.",
    "A group of toys come alive when their owner is away and face an existential threat.",
    "A lonely robot searches for signs of beauty on an abandoned future Earth.",
    "A widowed old man's house floats away on balloons, taking an unexpected passenger on an adventure.",
    "A clownfish crosses the entire ocean to find his missing son, with help from a forgetful friend.",
    "A blade runner tracks down and retires rogue replicants in a dystopian Los Angeles of 2049.",
    "Alien visitors communicate through a linguist whose encounter changes her perception of time.",
    "A young nobleman faces his destiny on a desert planet and leads a rebellion against oppressors.",
    "A warrior woman escapes a warlord by driving an armored war rig in a post-apocalyptic wasteland.",
    "A young man's visit to his girlfriend's family uncovers a horrifying secret at a remote estate.",
    "A family discovers they are being terrorized by their own doppelgangers on a beach vacation.",
    "A filmmaker investigates mysterious deaths and encounters extraterrestrial beings in rural California.",
    "A laundromat owner simultaneously lives countless parallel lives to stop a multiverse-threatening threat.",
    "A legendary hotel concierge and his devoted lobby boy navigate the decline of a grand European empire.",
    "Two children and a quirky scout troop share a magical island summer in the 1960s.",
    "A family of animals pull off a fantastic heist to save their beloved farm from greedy developers.",
    "A jazz drummer's obsessive pursuit of perfection under a terrifying instructor pushes him to extremes.",
    "A struggling pianist and a jazz musician fall in love while chasing their Hollywood dreams.",
    "An astronaut's remarkable story of courage, isolation, and the first steps on the Moon.",
    "A secret agent races to master a time-inversion algorithm that could prevent World War III.",
    "A man suffering from anterograde amnesia uses notes and tattoos to hunt his wife's killer.",
    "Two brilliant magicians become dangerous rivals, each trying to outdo the other's greatest trick.",
    "Allied soldiers are trapped on the beaches of France during a massive military evacuation in WWII.",
    "The remarkable and controversial story of J. Robert Oppenheimer, father of the atomic bomb.",
    "Barbie faces an existential crisis after a glitch sends her into the chaotic real world.",
    "A teen swings through a multiverse of spider-people to stop a reality-threatening villain.",
    "A jazz musician on the verge of a breakthrough experiences the beauty of life in a unique way.",
    "A Colombian family discovers that the magic of their enchanted home may be fading.",
    "A Polynesian chief's daughter sets sail across the ocean to save her people from a spreading darkness.",
    "A sharp-witted detective uses cunning and wit to unravel a poisoning mystery at a family gathering.",
    "A tech billionaire's murder sends a detective and his ragtag team on a globe-spanning investigation.",
    "A frontiersman miraculously survives a brutal bear attack and seeks justice from his traitorous guide.",
    "Two British soldiers race against time across dangerous no-man's-land during World War I.",
    "An ex-hitman goes on a bloody rampage of vengeance after assassins steal his car and kill his dog.",
]

OCCUPATIONS = [
    "Engineer", "Teacher", "Doctor", "Student", "Artist",
    "Scientist", "Lawyer", "Designer", "Manager", "Writer"
]


def generate_movies(n: int = 450) -> pd.DataFrame:
    """Generate a DataFrame of synthetic movies with n entries."""
    titles_pool = list(MOVIE_TITLES)
    # If we need more titles than available, generate extras
    while len(titles_pool) < n:
        prefixes = ["The", "A", "Beyond", "Dark", "Last", "Final", "Shadow", "Neon", "Iron", "Gold"]
        extras = [f"{random.choice(prefixes)} {t.split()[-1]} {random.randint(2, 9)}" for t in MOVIE_TITLES[:60]]
        titles_pool.extend(extras)
    titles_pool = titles_pool[:n]
    random.shuffle(titles_pool)

    movies = []
    for i, title in enumerate(titles_pool):
        num_genres = random.choices([1, 2, 3], weights=[0.2, 0.5, 0.3])[0]
        movie_genres = random.sample(ALL_GENRES, num_genres)
        cast = random.sample(ACTORS, 3)
        avg_r = round(np.clip(np.random.normal(6.8, 1.4), 1.0, 10.0), 1)
        votes = int(np.clip(np.random.lognormal(10, 1.3), 1000, 2_000_000))
        movie = {
            "movie_id": i + 1,
            "title": title,
            "genres": "|".join(movie_genres),
            "year": random.randint(1990, 2024),
            "avg_rating": avg_r,
            "votes": votes,
            "director": random.choice(DIRECTORS),
            "cast": "|".join(cast),
            "description": DESCRIPTIONS[i % len(DESCRIPTIONS)],
            "runtime_min": random.randint(80, 185),
            "language": random.choice(["English", "English", "English", "English", "Korean", "French", "Spanish", "Italian"]),
            "budget_million": round(random.uniform(1.0, 350.0), 1),
        }
        movies.append(movie)
    return pd.DataFrame(movies)


def generate_users(n: int = 200) -> pd.DataFrame:
    """Generate a DataFrame of synthetic users."""
    users = []
    for i in range(1, n + 1):
        num_genres = random.randint(2, 4)
        pref_genres = random.sample(ALL_GENRES, num_genres)
        user = {
            "user_id": i,
            "age": random.randint(16, 65),
            "occupation": random.choice(OCCUPATIONS),
            "preferred_genres": "|".join(pref_genres),
            "country": random.choice(["US", "UK", "CA", "AU", "IN", "DE", "FR", "KR", "JP"]),
        }
        users.append(user)
    return pd.DataFrame(users)


def generate_ratings(movies_df: pd.DataFrame, users_df: pd.DataFrame,
                     n: int = 4000) -> pd.DataFrame:
    """Generate a DataFrame of user-movie ratings."""
    movie_ids = movies_df["movie_id"].tolist()
    user_ids = users_df["user_id"].tolist()
    base_time = datetime(2019, 1, 1)

    # Give every user at least 8 ratings first
    ratings = []
    seen = set()
    for uid in user_ids:
        n_user = random.randint(8, 25)
        sampled = random.sample(movie_ids, min(n_user, len(movie_ids)))
        for mid in sampled:
            if (uid, mid) in seen:
                continue
            seen.add((uid, mid))
            movie_avg = movies_df.loc[movies_df["movie_id"] == mid, "avg_rating"].values[0]
            raw = movie_avg / 2 + random.gauss(0, 0.4)
            r = max(0.5, min(5.0, round(raw * 2) / 2))
            ts = base_time + timedelta(days=random.randint(0, 365 * 5), hours=random.randint(0, 23))
            ratings.append({"user_id": uid, "movie_id": mid, "rating": r,
                             "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S")})

    # Fill up to n
    attempts = 0
    while len(ratings) < n and attempts < n * 5:
        attempts += 1
        uid = random.choice(user_ids)
        mid = random.choice(movie_ids)
        if (uid, mid) in seen:
            continue
        seen.add((uid, mid))
        movie_avg = movies_df.loc[movies_df["movie_id"] == mid, "avg_rating"].values[0]
        raw = movie_avg / 2 + random.gauss(0, 0.4)
        r = max(0.5, min(5.0, round(raw * 2) / 2))
        ts = base_time + timedelta(days=random.randint(0, 365 * 5), hours=random.randint(0, 23))
        ratings.append({"user_id": uid, "movie_id": mid, "rating": r,
                         "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S")})

    return pd.DataFrame(ratings).reset_index(drop=True)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    print("🎬 Generating CineMatch Dataset (450 movies, 200 users, 4000+ ratings)...")

    movies_df = generate_movies(450)
    users_df = generate_users(200)
    ratings_df = generate_ratings(movies_df, users_df, 4000)

    movies_df.to_csv(os.path.join(out_dir, "movies.csv"), index=False)
    users_df.to_csv(os.path.join(out_dir, "users.csv"), index=False)
    ratings_df.to_csv(os.path.join(out_dir, "ratings.csv"), index=False)

    print(f"✅ movies.csv   → {len(movies_df)} movies")
    print(f"✅ users.csv    → {len(users_df)} users")
    print(f"✅ ratings.csv  → {len(ratings_df)} ratings")
    print(f"   Unique users rating: {ratings_df['user_id'].nunique()}")
    print(f"   Unique movies rated: {ratings_df['movie_id'].nunique()}")
    print("🎉 Dataset generation complete!")


if __name__ == "__main__":
    main()

