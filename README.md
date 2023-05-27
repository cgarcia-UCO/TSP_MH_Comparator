# TSP_MH_Comparator

This project is aimed at easily comparing different MHs on many problem instances (well, actually many _random uniformly
distributed TSP_ instances). Just add the MHs as functions in `mhs.py`, run

    python main.py

and follow the instructions.

Ahm, sorry! I forgot to say that MH functions must have explicitly two arguments:

- the former is the number of cities in the TSP instance, and
- the second is the function your MH has to use to evaluate solutions

_"But..., my MH has got several configuration parameters and..."_:

I am sorry, the framework is meant to compare MHs as a whole.
It is your responsibility to set the MH's parameters properly, so you expect a good behavior without knowing the caracteristics of the (_random uniformly distributed TSP_) problem instances.

"_But..., and the number of generations / evaluations / time budget / ...?_":

Ahm, yes! Please, replace your main loop with `while True:`.

"`while True:` ???":

Yes. The framework allows the user to stablish a maximum time budget for all the MHs. This is the only stopping condition considered. Afterwards, MHs are run until having consumed that time budget. How? Comments below.

## The inside

The framework creates a structure with the data of the TSP instances, and the results of the executed MHs, which is
stored incrementally in the file `data.pickle`.

When running `main.py`, the menu allows you to:

- Continue running your MHs on the TSP instances (you can ^C the execution and continue later!!)
- Remove the results of a MH on a selected TSP instance
- Generate and append a new TSP instance into the comparison framework (so MHs can be run on it subsequently or much later)
- Produce convergence and ranking evolution graphs of the executed MHs

If your executions run into errors, perhaps the file `log.log` helps us all to fix them!

About how the time budget is managed: The framework throws a specific exception when the time budget is consumed. Obviously, you should not catch that exception (but nevertheless, this is Python code, so you can dive into the code and make it 'useless', or not valid for its original purpose).

# Enjoy!
