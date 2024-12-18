from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc


class PokemonBrock(PokemonEnvironment):
    def __init__(
            self,
            act_freq: int,
            emulation_speed: int = 0,
            headless: bool = False,
    ) -> None:

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            #  WindowEvent.PRESS_BUTTON_START,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            # WindowEvent.RELEASE_BUTTON_START,
        ]

        super().__init__(
            act_freq=act_freq,
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )

    # store the visited coordinates.
    visited_coords = []
    seen = []
    # same_location_counter = 0
    episode_battle_wins = 0
    action = -1
    left_wall = -1
    right_wall = -1
    top_wall = -1
    bottom_wall = -1

    def _run_action_on_emulator(self, action_array):
        action = action_array[0]
        action = min(action, 0.99)

        # Continuous Action is a float between 0 - 1 from Value based methods
        # We need to convert this to an action that the emulator can understand
        bins = np.linspace(0, 1, len(self.valid_actions) + 1)
        button = np.digitize(action, bins) - 1
        self.action = button

        # Push the button for a few frames
        self.pyboy.send_input(self.valid_actions[button])

        for _ in range(self.act_freq):
            self.pyboy.tick()

        # Release the button
        self.pyboy.send_input(self.release_button[button])

    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here
        game_stats = self._generate_game_stats()
        self.get_wall_status()
        is_grass = self._is_grass_tile()
        battle = (self._read_m(0xD057) != 0x00)
        current_move = -1
        # ======================================Left Over=====================================
        # game_stats["seen_pokemon"],
        # game_stats["caught_pokemon"],
        # sum(game_stats["xp"]),
        # self.read_hp_as_a_fraction(),
        # len(self.seen),
        # len(self.visited_coords),
        # ======================================Left Over=====================================
        if battle:  # logic being that when in battle the state vector should not matter, we want it to do a set action
            map_array = self._get_screen_walkable_matrix()
            map_array = map_array.astype('int16')
            map_array[:] = -1
            fight_status_one_hot = np.zeros(4)
            fight_status_one_hot[self._get_fight_status()] = 1
            current_move = self._current_move()
            state_vector = np.concatenate((np.array([
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                battle,
                is_grass,
                self.read_enemy_hp_as_fraction(),  # normalise it
                self.action,
            ]), map_array.flatten(), fight_status_one_hot))

        else:  # traversing
            fight_status_one_hot = np.ones(4) * -1
            state_vector = np.concatenate((np.array([
                game_stats["location"]["x"],
                game_stats["location"]["y"],
                game_stats["location"]["map_id"],
                # obstacles?
                self.top_wall,
                self.bottom_wall,
                self.right_wall,
                self.left_wall,
                battle,
                is_grass,
                -1,
                self.action,
            ]), self._get_screen_walkable_matrix().flatten(), fight_status_one_hot))

        return state_vector

    def get_wall_status(self):
        map_data = self._get_screen_walkable_matrix()
        # Initialize direction flags
        walkable_left = 0
        walkable_right = 0
        walkable_up = 0
        walkable_down = 0
        player_x = 4
        player_y = 4

        # Check if the tile to the left is walkable
        if player_x > 0 and map_data[player_y][player_x - 1] == 1:
            walkable_left = 1

        # Check if the tile to the right is walkable
        if player_x < len(map_data[0]) - 1 and map_data[player_y][player_x + 1] == 1:
            walkable_right = 1

        # Check if the tile above is walkable
        if player_y > 0 and map_data[player_y - 1][player_x] == 1:
            walkable_up = 1

        # Check if the tile below is walkable
        if player_y < len(map_data) - 1 and map_data[player_y + 1][player_x] == 1:
            walkable_down = 1

        self.left_wall = walkable_left
        self.right_wall = walkable_right
        self.bottom_wall = walkable_down
        self.top_wall = walkable_up

    def penalty_walls(self):
        battle = (self._read_m(0xD057) != 0x00)
        penalty = 0
        if battle:
            return 0
        else:
            if self.left_wall == 0:
                penalty += -0.1
            if self.right_wall == 0:
                penalty += -0.1
            if self.top_wall == 0:
                penalty += -0.1
            if self.bottom_wall == 0:
                penalty += -0.1
        return penalty

    def read_enemy_hp_as_fraction(self) -> float:
        enemy_max = self._read_m(0xCFF5)
        max_hp = max(enemy_max, 1)
        return self.enemy_hp / max_hp

    def read_hp_as_a_fraction(self) -> float:
        current_and_max_hp = self._read_party_hp()
        current_hp, max_hp = current_and_max_hp["current"], current_and_max_hp["max"]

        current_hp = sum(current_hp)
        max_hp = sum(max_hp)
        # make sure we don't divide by 0
        max_hp = max(max_hp, 1)

        return current_hp / max_hp

    prior_enemy_hp = -1
    enemy_hp = -1

    def update_enemy_hp(self) -> None:
        enemy_hp = self._read_m(0xCFE7)
        self.enemy_hp = enemy_hp

    def battle_reward(self):
        self.prior_enemy_hp = self.enemy_hp
        self.update_enemy_hp()
        enemy_max = self._read_m(0xCFF5)
        if self.enemy_hp < self.prior_enemy_hp:
            print(
                f"We engaged in mortal kombat: {self.prior_enemy_hp} - {self.enemy_hp} = {((self.prior_enemy_hp - self.enemy_hp) * 20)}")
            if enemy_max > self.prior_enemy_hp:  # if we previously battled give a bigger reward
                print("consecutive battle: ", ((self.prior_enemy_hp - self.enemy_hp) * 20) * 1.5)
                return ((self.prior_enemy_hp - self.enemy_hp) * 20) * 1.5
            return (self.prior_enemy_hp - self.enemy_hp) * 20
        return 0

    def has_won(self, new_state):
        if self.enemy_hp == 0 and self.prior_enemy_hp != 0:
            print(f"KILLED AN ENEMY + 300")
            return 300
        return 0

    def _get_fight_status(self):
        return self._read_m(0xCC26)

    def _current_move(self):
        return self._read_m(0xCC29)

    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here
        temp_reward = 0
        battle_active = (self._read_m(0xD057) != 0x00)

        # exploration rewards:
        if not battle_active:
            location = new_state.get("location")
            temp_reward += self.exploration_reward(location)
            temp_reward += self.penalty_walls()

        # battle rewards
        if battle_active:
            temp_reward += -0.5  # penalise idle battle state
            temp_reward += self.battle_reward()
        else:
            self.enemy_hp = -1
            self.prior_enemy_hp = -1
        temp_reward += self.has_won(new_state)

        return temp_reward

    def exploration_reward(self, location):
        key = f"{location}"
        reward = -1

        # if self._is_grass_tile():
        #     reward += 0.5
        # #
        # if key in self.seen and self._is_grass_tile():  # only penalise staying in the same place if its grass and not battle
        #     reward += -1

        if location["map_id"] not in self.visited_coords:
            self.visited_coords.append(location["map_id"])
            bruh = location["map_id"]
            print(f"new location!: {bruh}")
            if bruh != 40:
                reward += 600

        # if self.prior_game_stats["location"]["x"] != location["x"] and self.prior_game_stats["location"]["y"] != \
        #         location["y"]:
        #     reward += -1

        if self.prior_game_stats["location"]["x"] == location["x"] or self.prior_game_stats["location"]["y"] == \
                location["y"]:
            reward += 0.5

        if location["map_id"] == 12:
            if self.prior_game_stats["location"]["y"] > location["y"] and self.prior_game_stats["location"]["map_id"] == \
                    location["map_id"] and key not in self.seen:
                distance_to_goal = abs(0 - location["y"])
                reward += 2 + (50 / (distance_to_goal + 1))
                print("Reward for area 12: ", 2 + (50 / (distance_to_goal + 1)))

                y = self.prior_game_stats["location"]["y"]
                x = self.prior_game_stats["location"]["x"]
                print(f"prior location: {x},{y}")
                print(f"current:{location}")

        elif self.prior_game_stats["location"]["y"] > location["y"] and self.prior_game_stats["location"]["map_id"] == \
                location["map_id"] and key not in self.seen:
            distance_to_goal = abs(0 - location["y"])
            reward += 1 + (50 / (distance_to_goal + 1))

        # if unseen then reward it and add it
        if key not in self.seen:
            self.seen.append(key)
            reward += 1

        return reward

    # recent_locations=[]
    # def exploration_reward(self, location):
    #     key = f"{location}"
    #     reward = 0
    #
    #     # Reward for visiting new locations
    #     if location["map_id"] not in self.visited_coords:
    #         self.visited_coords.append(location["map_id"])
    #         distance_bonus = 1 / (len(self.visited_coords) + 1)
    #         reward += 100 * distance_bonus
    #         print(f"New location! Scaled reward: {reward}")
    #
    #     # Penalize for revisiting recent locations
    #     self.recent_locations.append(location)
    #     if len(self.recent_locations) > 5:
    #         self.recent_locations.pop(0)
    #
    #     if location in self.recent_locations:
    #         reward -= 5
    #         print("Penalized for revisiting recent location.")
    #
    #     # Reward for moving toward the goal in map_id 12
    #     if location["map_id"] == 12:
    #         if self.prior_game_stats["location"]["y"] > location["y"]:
    #             distance_to_goal = abs(0 - location["y"])
    #             reward += 2 + (10 / (distance_to_goal + 1))
    #             print(f"Reward for moving toward goal in map 12: {reward}")
    #
    #     # Gradient reward for progression toward a general target
    #     target_y, target_x = 0, 0  # Example target
    #     distance_to_target = abs(target_y - location["y"]) + abs(target_x - location["x"])
    #     reward += 10 / (distance_to_target + 1)
    #     print(f"Reward for progression toward target: {reward}")
    #
    #     # Small reward for visiting an unseen location
    #     if key not in self.seen:
    #         self.seen.append(key)
    #         reward += 1
    #
    #     # Clip the reward to prevent instability
    #     reward = np.clip(reward, -10, 10)
    #
    #     return reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here
        if self.steps >= 5000:
            self.visited_coords.clear()
            self.seen.clear()
            self.enemy_hp = -1
            self.prior_enemy_hp = -1
            self.action = -1
            self.left_wall = 0
            self.right_wall = 0
            self.top_wall = 0
            self.bottom_wall = 0
            # self.recent_locations.clear()
            return True
        return False
