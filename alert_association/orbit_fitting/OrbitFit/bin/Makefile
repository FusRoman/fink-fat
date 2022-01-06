simple_orbfit : orbfit.x 2015AB.fel

2015AB.fel : 	orbfit.x
		-(ln -s ../tests/bineph/testout/AST17.bai_431_fcct AST17.bai)
		-(ln -s ../tests/bineph/testout/AST17.bep_431_fcct AST17.bep)
		-(cp -p allobs/*.obs mpcobs/)
		-(cp -p allobs/*.oop .)
		-(cp -p allobs/*.inp .)
		time ./orbfit.x < test2.inp
		
distclean: clean

clean:
	-(rm -f *.pro *.clo *.odc *.oel *.olg *.oep *.rwo *~ reject.rep *.err)
	-(rm -f *.eq0 *.oop *.obs *.bat *.rad *.ele *.inp)
	-(rm -f *.bai *bep)
	-(rm -f mpcobs/*)


