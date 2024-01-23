set DIR_WITH_FRAMS_LIBRARY="D:\Projekty\STUDIA MAGISTERSKIE\SEM 2 - AIMIB\aimib2\Framsticks50rc24"
@REM set DIR_WITH_FRAMS_LIBRARY="C:\Users\natal\Desktop\aimib\Framsticks50rc24"


for /L %%S in (1,1,12) do (
    start /B python main_gp.py ^
        -path %DIR_WITH_FRAMS_LIBRARY%  ^
        -sim eval-allcriteria.sim;deterministic.sim;sample-period-2.sim ^
        -opt vertpos ^
        -max_numparts 15 ^
        -max_numjoints 30 ^
        -max_numneurons 20 ^
        -max_numconnections 30 ^
        -genformat 1 ^
        -popsize 60 ^
        -generations 130 ^
        -hof_size 1 ^
        -series %%S
)

for /L %%S in (1,1,12) do (
    start /B python main_fram.py ^
        -path %DIR_WITH_FRAMS_LIBRARY%  ^
        -sim eval-allcriteria.sim;deterministic.sim;sample-period-2.sim ^
        -opt vertpos ^
        -max_numparts 15 ^
        -max_numjoints 30 ^
        -max_numneurons 20 ^
        -max_numconnections 30 ^
        -genformat 1 ^
        -popsize 60 ^
        -generations 130 ^
        -hof_size 1 ^
        -series %%S
)