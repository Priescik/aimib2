@REM set DIR_WITH_FRAMS_LIBRARY="D:\Projekty\STUDIA MAGISTERSKIE\SEM 2 - AIMIB\Framsticks50rc24"
set DIR_WITH_FRAMS_LIBRARY="D:\Projekty\STUDIA MAGISTERSKIE\SEM 2 - AIMIB\aimib2\Framsticks50rc24"

@REM for %%E in (100) do (
for %%E in (100, 200, 500, 1000) do (
    @REM for /L %%S in (1,1,1) do (
    for /L %%S in (1,1,10) do (
        python main_fram.py ^
            -path %DIR_WITH_FRAMS_LIBRARY%  ^
            -sim eval-allcriteria-mini.sim;deterministic.sim;sample-period-2.sim;zero-grav.sim;energy%%E.sim;recording-body-coords.sim  ^
            -opt vertpos ^
            -max_numparts 30 ^
            -max_numgenochars 50 ^
            -genformat 1 ^
            -popsize 50 ^
            -generations 100 ^
            -hof_size 1 ^
            -series %%S
            @REM -initialgenotype XXXXX
            @REM -hof_savefile HoFs/HoF-f9-%%M-%%N.gen ^
            @REM -initialgenotype (CXX,XXXmXXXCX[G]X(XX),cXX[S]QIXXXXXX)
    )
)

@REM python FramsticksEvolution.py -path %DIR_WITH_FRAMS_LIBRARY%  -sim eval-allcriteria.sim;deterministic.sim;sample-period-2.sim  -opt vertpos -max_numparts 30 -max_numgenochars 50 -initialgenotype /*9*/BLU   -popsize 50    -generations 5 -hof_size 1
