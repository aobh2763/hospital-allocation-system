"""Block 4: resource allocation by state-space search (Uniform Cost Search, A*)."""

from search.algorithms import Decision, SearchResult, allocate, allocation, check_allocation, diagnose, search
from search.problem import AllocationProblem, from_requirements

__all__ = ["AllocationProblem", "Decision", "SearchResult", "allocate", "allocation", "check_allocation", "diagnose",
           "from_requirements", "search"]
