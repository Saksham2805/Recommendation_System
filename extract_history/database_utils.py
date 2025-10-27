import sqlite3
import os
from typing import Optional, Dict, Any

class MovieDatabase:
    """
    Manages SQLite database for movie details to avoid repeated API calls.
    """

    def __init__(self, db_path: str = "movies_details_db/movies.db"):
        """
        Initialize the movie database.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        """
        Create the movies table if it doesn't exist.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS movies (
                    title TEXT PRIMARY KEY,
                    genre TEXT NOT NULL,
                    description TEXT NOT NULL
                )
            ''')
            conn.commit()

    def get_movie_details(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Get movie details from database by title.

        Args:
            title: The movie title to search for.

        Returns:
            Dictionary with 'genre' and 'description' if found, None otherwise.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT genre, description FROM movies WHERE title = ?', (title,))
            row = cursor.fetchone()

            if row:
                return {
                    'genre': row[0],
                    'description': row[1]
                }
            return None

    def add_movie_details(self, title: str, genre: str, description: str):
        """
        Add movie details to the database.

        Args:
            title: The movie title.
            genre: The movie genre.
            description: The movie description.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO movies (title, genre, description)
                VALUES (?, ?, ?)
            ''', (title, genre, description))
            conn.commit()

    def get_all_movies(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all movies from the database.

        Returns:
            Dictionary with title as key and details as value.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT title, genre, description FROM movies')
            rows = cursor.fetchall()

            movies = {}
            for row in rows:
                movies[row[0]] = {
                    'genre': row[1],
                    'description': row[2]
                }
            return movies

    def clear_database(self):
        """
        Clear all entries from the database. Use with caution.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM movies')
            conn.commit()

# Example usage
if __name__ == "__main__":
    db = MovieDatabase()

    # Test adding and retrieving a movie
    db.add_movie_details("Inception", "Sci-Fi", "A thief who steals corporate secrets through the use of dream-sharing technology.")

    # Retrieve it
    details = db.get_movie_details("Inception")
    print(details)  # Should print the details
