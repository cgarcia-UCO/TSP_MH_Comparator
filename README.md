# TSP_MH_Comparator

This project is aimed at easily comparing different MHs on many problem instances (well, actually many _random uniformly
distributed TSP_ instances). Just add the MHs as functions in `mhs.py`, run

    python main.py

and follow the instructions.

Ahm, sorry! I forgot to say that MH functions must have explicitly two arguments:
 - the former is the number of cities in the TSP instance, and
 - the second is the function your MH has to use to evaluate solutions

_"But..., my MH has got several configuration parameters and..."_: I am sorry, the framework is meant to compare MHs as a whole.
It is your responsibility to set the MH's parameters properly, so you expect a good behavior without knowing the caracteristics
of the (_random uniformly distributed TSP_) problem instances.


## The inside

The framework creates a structure with the data of the TSP instances, and the results of the executed MHs, which is
stored incrementally in the file `data.pickle`.

When running `main.py`, the menu allows you to:
 - Continue running your MHs on the TSP instances (you can ^C the execution and continue later!!)
 - Remove the results of a MH on a selected TSP instance
 - Generate and append a new TSP instance into the comparison framework (so MHs can be run on it subsequently or much later)
 - Produce convergence and ranking evolution graphs of the executed MHs

If your executions run into errors, perhaps the file `log.log` helps us all to fix them!

# Enjoy!