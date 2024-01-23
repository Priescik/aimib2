set DIR_WITH_FRAMS_LIBRARY="D:\Projekty\STUDIA MAGISTERSKIE\SEM 2 - AIMIB\Framsticks50rc24"
set DIR_WITH_FRAMS_LIBRARY="C:\Users\natal\Desktop\aimib\Framsticks50rc24"

for %%P in (4) do (
    for /L %%N in (1,1,10) do (
        start /B python FramsticksEvolution_lab3_dynamic.py ^
            -path %DIR_WITH_FRAMS_LIBRARY%  ^
            -sim eval-allcriteria.sim;deterministic.sim;sample-period-longest.sim;wlasne-prawd-%%P.sim ^
            -opt velocity ^
            -max_numparts 15 ^
            -max_numjoints 30 ^
            -max_numneurons 20 ^
            -max_numconnections 30 ^
            -genformat 1 ^
            -pxov 0 ^
            -popsize 50 ^
            -generations 200 ^
            -hof_size 1 ^
            -hof_savefile HoFs/HoF-vel-prawd%%P-%%N.gen ^
            -series %%P-%%N
    )
)