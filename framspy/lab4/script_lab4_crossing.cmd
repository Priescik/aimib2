set DIR_WITH_FRAMS_LIBRARY="D:\Projekty\STUDIA MAGISTERSKIE\SEM 2 - AIMIB\aimib2\Framsticks50rc24"
@REM set DIR_WITH_FRAMS_LIBRARY="C:\Users\natal\Desktop\aimib\Framsticks50rc24"


start /B python FramsticksEvolution_lab4_crossover.py ^
    -path %DIR_WITH_FRAMS_LIBRARY%  ^
    -sim eval-allcriteria.sim;deterministic.sim;sample-period-longest.sim ^
    -opt velocity ^
    -max_numparts 15 ^
    -max_numjoints 30 ^
    -max_numneurons 20 ^
    -max_numconnections 30 ^
    -popsize 1 ^
    -generations 1 ^
    -hof_size 1 ^