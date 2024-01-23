set DIR_WITH_FRAMS_LIBRARY="D:\Projekty\STUDIA MAGISTERSKIE\SEM 2 - AIMIB\Framsticks50rc24"
@REM set DIR_WITH_FRAMS_LIBRARY="C:\Users\natal\Desktop\aimib\Framsticks50rc24"

            @REM -pxov 0 ^ ???

@REM for %%F in ("0", "1", "4", "H") do (
for %%F in ("4") do (
    for /L %%N in (1,1,10) do (
@REM     for /L %%N in (1,1,1) do (
        start /B python FramsticksEvolution_lab4.py ^
            -path %DIR_WITH_FRAMS_LIBRARY%  ^
            -sim eval-allcriteria.sim;deterministic.sim;sample-period-longest.sim ^
            -opt velocity ^
            -max_numparts 15 ^
            -genformat %%F   ^
            -max_numjoints 30 ^
            -max_numneurons 20 ^
            -max_numconnections 30 ^
            -popsize 40 ^
            -generations 80 ^
            -hof_size 100 ^
            -hof_savefile HoFs/HoF-f%%F-%%N.gen ^
            -series %%F-%%N
    )
)