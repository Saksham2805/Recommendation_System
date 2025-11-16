import os
import csv
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from extract_history.database_utils import MovieDatabase

class MovieDetailsFetcher:
    """
    Fetches genre and description for movies using Google Generative AI.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the MovieDetailsFetcher.

        Args:
            api_key: Google AI API key. If None, loads from environment.
        """
        load_dotenv()

        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')

        if not api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY in .env or pass directly.")

        genai.configure(api_key=api_key)

        # Initialize the model
        try:
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini model: {e}")

        # Initialize the movie database
        self.db = MovieDatabase()

    def read_movie_titles_from_csv(self, csv_file_path: str) -> List[Dict[str, Any]]:
        """
        Read movie titles and counts from a CSV file.

        Args:
            csv_file_path: Path to the CSV file.

        Returns:
            List of dictionaries with 'title' and 'count'.
        """
        movies = []
        try:
            with open(csv_file_path, mode='r', encoding='utf-8-sig') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    title = row.get('title', '').strip().strip("'\"")  # Remove surrounding quotes
                    count_str = row.get('count', '0').strip()
                    if title:
                        try:
                            count = int(count_str)
                        except ValueError:
                            count = 0
                        movies.append({
                            'title': title,
                            'count': count
                        })
        except Exception as e:
            print(f"Error reading CSV file {csv_file_path}: {e}")
            return []
        print(f"Read {len(movies)} movies from {csv_file_path}")
        if movies:
            print(f"First movie: {movies[0]}")
        return movies

    def get_movie_details(self, movie_title: str) -> Dict[str, Any]:
        """
        Fetch genre and description for a movie. First checks database, then API if needed.

        Args:
            movie_title: The title of the movie.

        Returns:
            Dictionary with 'genre' and 'description'.
        """
        # First check if movie is already in database
        cached_details = self.db.get_movie_details(movie_title)
        if cached_details:
            print(f"  Using cached details for '{movie_title}'")
            return cached_details

        # Not in database, fetch from API
        print(f"  Fetching details from API for '{movie_title}'")

        prompt = f"""
Provide detailed information about the movie or TV show titled "{movie_title}".

Respond ONLY with a valid JSON object in this exact format:
{{
    "genre": "the primary genre of the movie/TV show (e.g., Action, Comedy, Drama, Horror, Romance, etc.)",
    "description": "a brief 2-3 sentence description of the plot without spoilers"
}}

Do not include any other text, explanations, or formatting. Just the JSON object.
        """

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            # Find JSON in response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]

                # Clean JSON
                json_str = json_str.replace('\n', '').replace('\r', '').strip()

                details = json.loads(json_str)

                # Validate
                genre = details.get('genre', '').strip()
                description = details.get('description', '').strip()

                # Cache in database
                self.db.add_movie_details(movie_title, genre, description)

                return {
                    'genre': genre,
                    'description': description
                }
            else:
                details = self._fallback_details(movie_title)
                # Cache fallback details too
                self.db.add_movie_details(movie_title, details['genre'], details['description'])
                return details

        except Exception as e:
            print(f"Error fetching details for {movie_title}: {e}")
            details = self._fallback_details(movie_title)
            # Cache fallback details too
            self.db.add_movie_details(movie_title, details['genre'], details['description'])
            return details

    def _fallback_details(self, movie_title: str) -> Dict[str, Any]:
        """
        Fallback details if API fails.

        Args:
            movie_title: The title of the movie.

        Returns:
            Dictionary with default 'genre' and 'description'.
        """
        return {
            'genre': "Unknown",
            'description': f"A movie titled {movie_title}."
        }

    def process_csv_and_save(self, csv_file_path: str, output_file_path: str, delay: float = 1.0):
        """
        Process a CSV file, fetch details for each movie, and save enriched data.

        Args:
            csv_file_path: Path to input CSV file.
            output_file_path: Path to output file (JSON format).
            delay: Delay between API calls in seconds to avoid rate limits.
        """
        movies = self.read_movie_titles_from_csv(csv_file_path)
        enriched_data = []

        print(f"Processing {len(movies)} movies from {csv_file_path}...")

        for i, movie in enumerate(movies, 1):
            print(f"Fetching details for: {movie['title']} ({i}/{len(movies)})")

            details = self.get_movie_details(movie['title'])

            enriched_movie = {
                'title': movie['title'],
                'count': movie['count'],
                'genre': details['genre'],
                'description': details['description']
            }

            enriched_data.append(enriched_movie)

            # Respect rate limits
            if i < len(movies):
                time.sleep(delay)

        # Save to file
        try:
            with open(output_file_path, mode='w', encoding='utf-8') as file:
                json.dump(enriched_data, file, indent=4, ensure_ascii=False)
            print(f"Enriched data saved to {output_file_path}")
        except Exception as e:
            print(f"Error saving enriched data to {output_file_path}: {e}")

def main():
    """
    Main function to process all CSV files in histories directory.
    """
    fetcher = MovieDetailsFetcher()

    histories_dir = 'histories'
    cleaned_dir = 'histories/detailed'

    os.makedirs(cleaned_dir, exist_ok=True)

    for csv_file in os.listdir(histories_dir):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(histories_dir, csv_file)
            base_name = os.path.splitext(csv_file)[0]
            output_path = os.path.join(cleaned_dir, f"{base_name}_enriched.json")

            fetcher.process_csv_and_save(csv_path, output_path)

if __name__ == "__main__":
    main()
