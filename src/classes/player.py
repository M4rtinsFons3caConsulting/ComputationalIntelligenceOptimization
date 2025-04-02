# Each player is defined by the following attributes: 
#  ●   Skill rating: Represents the player's ability. 
#  ●   Cost: The player's salary. 
#  ●   Position:  One  of  four  roles:  Goalkeeper  (GK),  Defender  (DEF),  Midfielder  (MID),  or Forward (FWD).

class Player:
    def __init__(self, skill, salary, position):
        self.skill = skill
        self.salary = salary
        self.position = position


class Team:
    def __init__(self):
        self.cash = 750
        self.roles = ['GK', 'DEF', 'DEF', 'MID', 'MID', 'FWD', 'FWD']

    get_player




# We can draft each team at a time,
# We can draft each position at a time,
# oH OH , i got it

# We need to have some random starting setup, that is valid
# Then we just move pair wise between any two teams, ofc easiset option
# To speed it up we should probably create a tabu system, that checks if ignoring the team numbers, a specific ordering had been done
# 
#

# added this here for now
class Solver:
    def __init__(self, num_elements: int):
        if num_elements > 64:
            raise ValueError("Bitmask approach only works for up to 64 elements.")
        self.num_elements = num_elements
        self.seen_combinations = set()
    
    def encode(self, elements: list[int]) -> int:
        """Generate a unique bitmask for a given set of elements."""
        bitmask = 0
        for elem in elements:
            if elem < 0 or elem >= self.num_elements:
                raise ValueError(f"Element {elem} is out of range (0-{self.num_elements-1}).")
            bitmask |= (1 << elem)
        return bitmask
    
    def is_new_combination(self, elements: list[int]) -> bool:
        """Check if this combination is new, and store it if so."""
        bitmask = self.encode(elements)
        if bitmask in self.seen_combinations:
            return False
        self.seen_combinations.add(bitmask)
        return True

# Example usage:
solver = Solver(num_elements=35)
print(solver.is_new_combination([1, 3, 5]))  # True (new combination)
print(solver.is_new_combination([5, 3, 1]))  # False (same as previous)
print(solver.is_new_combination([2, 4, 6]))  # True (new combination)

# Id say the cash one is simple to detect just check if the cash any one team spent after
#  a mutation is larger than 750, if that happens just rerool the mutaiton
# 
# after getting a player, just subtract their cost from cash, if cash goes below zero, reroll?




