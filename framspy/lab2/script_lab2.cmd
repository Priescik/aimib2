set DIR_WITH_FRAMS_LIBRARY="D:\Projekty\STUDIA MAGISTERSKIE\SEM 2 - AIMIB\Framsticks50rc24"

@REM for %%M in (0,005,010,020,030,040,050) do (
@REM     for /L %%N in (1,1,10) do (
@REM         start /B python FramsticksEvolution.py ^
@REM             -path %DIR_WITH_FRAMS_LIBRARY%  ^
@REM             -sim eval-allcriteria.sim;deterministic.sim;sample-period-2.sim;f9-mut-%%M.sim ^
@REM             -opt vertpos ^
@REM             -max_numparts 30 ^
@REM             -max_numgenochars 50 ^
@REM             -initialgenotype /*9*/BLU ^
@REM             -popsize 50 ^
@REM             -generations 100 ^
@REM             -hof_size 1 ^
@REM             -hof_savefile HoFs/HoF-f9-%%M-%%N.gen ^
@REM             -series %%N
@REM     )
@REM )

for %%F in (0,1,4,9) do (
    for /L %%N in (1,1,10) do (
        start /B python FramsticksEvolution2.py ^
            -path %DIR_WITH_FRAMS_LIBRARY%  ^
            -sim eval-allcriteria.sim;deterministic.sim;sample-period-2.sim;only-body.sim ^
            -opt vertpos ^
            -max_numparts 30   ^
            -genformat %%F   ^
            -popsize 50    ^
            -generations 200 ^
            -hof_size 1 ^
            -hof_savefile HoFs2/HoF-f%%F-%%N.gen ^
            -series %%N
    )
)