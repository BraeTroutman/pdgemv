import Data.List

main :: IO ()
main = (print . head) bests

data Mtx = Mtx [[Bool]]

instance Show Mtx where
  show (Mtx m) = unlines $ map (map (\x -> if x then 'X' else '-')) m

-- our input matrix
mtx :: Mtx
mtx = Mtx [[True,False,False,False,True,True,False,False,True,False,False,False],
           [False,True,False,False,True,False,False,False,False,False,False,True],
           [False,False,True,True,False,False,False,False,False,False,True,False],
           [False,False,False,True,False,False,False,False,False,True,True,True],
           [False,True,False,False,True,False,False,False,True,False,False,True],
           [False,False,False,False,False,True,False,False,True,False,False,False],
           [True,False,False,True,True,True,True,False,False,True,False,False],
           [False,True,False,False,True,False,False,True,False,False,False,False],
           [True,False,False,False,False,False,True,False,True,False,False,False],
           [False,False,True,True,False,False,False,False,False,True,False,False],
           [False,False,True,True,False,False,False,False,True,False,True,True],
           [True,False,False,True,False,False,False,True,True,False,False,True]]

unbox :: Mtx -> [[Bool]]
unbox (Mtx m) = m

perms :: [Mtx]
perms = map Mtx $ permutations $ unbox mtx

groupProcs (Mtx m) = (take 4 m, take 4 $ drop 4 m, drop 8 m)

zip4List :: [[a]] -> [(a,a,a,a)]
zip4List [(a:as), (b:bs), (c:cs), (d:ds)] = (a,b,c,d):zip4List [as,bs,cs,ds]
zip4List [_,_,_,_] = []
zip4List [] = []

emptyCol :: (Bool,Bool,Bool,Bool) -> Bool
emptyCol (a,b,c,d) = not $ a || b || c || d

colscore m = (length $ filter id $ map emptyCol $ zip4List one,
            length $ filter id $ map emptyCol $ zip4List two,
            length $ filter id $ map emptyCol $ zip4List three)
  where (one,two,three) = groupProcs m

countscore m = (sum $ map (length . filter id) one,
              sum $ map (length . filter id) two,
              sum $ map (length . filter id) three)
  where (one,two,three) = groupProcs m

--extract the first 100000 permutation where the nonzeroes are evenly distributed between processes
evenDists = map snd $ take 100000 $ filter (\x -> (fst x) == (15,15,15)) $ map (\m -> (countscore m, m)) perms

--extract the permutations from evenDists that have 4 empty columns per processor
bests = map snd $ filter (\x -> (fst x) == (3,4,4)) $ map (\m -> (colscore m, m)) evenDists

