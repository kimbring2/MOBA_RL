import numpy as numpy

def get_obs_entity(obs):
    Hitpoints= obs[0]
    Ability0Ready= obs[1]
    FriendStatueDistance= obs[2]
    FriendStatueAngle= obs[3]
    Friend1Distance= obs[4]
    Friend1Angle= obs[5]
    Friend2Distance= obs[6]
    Friend2Angle= obs[7]
    EnemyStatueDistance= obs[8]
    EnemyStatueAngle= obs[9]
    Enemy1Distance= obs[10]
    Enemy1Angle= obs[11]
    Enemy2Distance= obs[12]
    Enemy2Angle= obs[13]
    Enemy3Distance= obs[14]
    Enemy3Angle= obs[15]
    HasFocus= obs[16]
    FocusRelativeRotation= obs[17]
    FocusFacingUs= obs[18]
    FocusFocusingBack= obs[19]
    FocusHitpoints= obs[20]
    Ability1Ready= obs[21]
    Ability2Ready= obs[22]
    FocusDazed= obs[23]
    FocusCrippled= obs[24]
    HeightFront1= obs[25]
    HeightFront5= obs[26]
    HeightBack2= obs[27]
    PositionLeftRight= obs[28]
    PositionUpDown= obs[29]
    Stuck= obs[30]
    UnusedSense31= obs[31]
    HasTalons= obs[32]
    HasBloodClaws= obs[33]
    HasCleavers= obs[34]
    HasCripplers= obs[35]
    HasHealingGland= obs[36]
    HasVampireGland= obs[37]
    HasFrogLegs= obs[38]
    HasPistol= obs[39]
    HasMagnum= obs[40]
    HasBlaster= obs[41]
    HasParalyzingDart= obs[42]
    HasIronBubblegum= obs[43]
    HasHeliumBubblegum= obs[44]
    HasShell= obs[45]
    HasTrombone= obs[46]
    FocusHasTalons= obs[47]
    FocusHasBloodClaws= obs[48]
    FocusHasCleavers= obs[49]
    FocusHasCripplers= obs[50]
    FocusHasHealingGland= obs[51]
    FocusHasVampireGland= obs[52]
    FocusHasFrogLegs= obs[53]
    FocusHasPistol= obs[54]
    FocusHasMagnum= obs[55]
    FocusHasBlaster= obs[56]
    FocusHasParalyzingDart= obs[57]
    FocusHasIronBubblegum= obs[58]
    FocusHasHeliumBubblegum= obs[59]
    FocusHasShell= obs[60]
    FocusHasTrombone= obs[61]
    UnusedExtraSense30= obs[62]
    UnusedExtraSense31= obs[63]
    
    print("Hitpoints: " + str(Hitpoints))
    print("Ability0Ready: " + str(Ability0Ready))

    print("FriendStatueDistance: " + str(FriendStatueDistance))
    print("FriendStatueAngle: " + str(FriendStatueAngle))
    print("Friend1Distance: " + str(Friend1Distance))
    print("Friend1Angle: " + str(Friend1Angle))
    print("Friend2Angle: " + str(Friend2Angle))
    print("Friend2Distance: " + str(Friend2Distance))
    print("Friend2Angle: " + str(Friend2Angle))

    print("EnemyStatueDistance: " + str(EnemyStatueDistance))
    print("EnemyStatueAngle: " + str(EnemyStatueAngle))
    print("Enemy1Distance: " + str(Enemy1Distance))
    print("Enemy1Angle: " + str(Enemy1Angle))
    print("Enemy2Distance: " + str(Enemy2Distance))
    print("Enemy2Angle: " + str(Enemy2Angle))
    print("Enemy3Distance: " + str(Enemy3Distance))
    print("Enemy3Angle: " + str(Enemy3Angle))

    print("HasFocus: " + str(HasFocus))
    print("FocusRelativeRotation: " + str(FocusRelativeRotation))
    print("FocusFacingUs: " + str(FocusFacingUs))
    print("FocusFocusingBack: " + str(FocusFocusingBack))
    print("FocusHitpoints: " + str(FocusHitpoints))
    print("Ability1Ready: " + str(Ability1Ready))
    print("Ability2Ready: " + str(Ability2Ready))
    print("FocusDazed: " + str(FocusDazed))
    print("FocusCrippled: " + str(FocusCrippled))
    print("HeightFront1: " + str(HeightFront1))
    print("HeightFront5: " + str(HeightFront5))
    print("HeightBack2: " + str(HeightBack2))
    print("PositionLeftRight: " + str(PositionLeftRight))
    print("PositionUpDown: " + str(PositionUpDown))
    print("Stuck: " + str(Stuck))

    print("UnusedSense31: " + str(UnusedSense31))

    print("HasTalons: " + str(HasTalons))
    print("HasBloodClaws: " + str(HasBloodClaws))
    print("HasCleavers: " + str(HasCleavers))
    print("HasCripplers: " + str(HasCripplers))
    print("HasHealingGland: " + str(HasHealingGland))
    print("HasVampireGland: " + str(HasVampireGland))
    print("HasFrogLegs: " + str(HasFrogLegs))
    print("HasPistol: " + str(HasPistol))
    print("HasMagnum: " + str(HasMagnum))
    print("HasBlaster: " + str(HasBlaster))
    print("HasParalyzingDart: " + str(HasParalyzingDart))
    print("HasIronBubblegum: " + str(HasIronBubblegum))
    print("HasHeliumBubblegum: " + str(HasHeliumBubblegum))
    print("HasShell: " + str(HasShell))
    print("HasTrombone: " + str(HasTrombone))

    print("FocusHasTalons: " + str(FocusHasTalons))
    print("FocusHasBloodClaws: " + str(FocusHasBloodClaws))
    print("FocusHasCleavers: " + str(FocusHasCleavers))
    print("FocusHasCripplers: " + str(FocusHasCripplers))
    print("FocusHasHealingGland: " + str(FocusHasHealingGland))
    print("FocusHasVampireGland: " + str(FocusHasVampireGland))
    print("FocusHasFrogLegs: " + str(FocusHasFrogLegs))
    print("FocusHasPistol: " + str(FocusHasPistol))
    print("FocusHasMagnum: " + str(FocusHasMagnum))
    print("FocusHasBlaster: " + str(FocusHasBlaster))
    print("FocusHasParalyzingDart: " + str(FocusHasParalyzingDart))
    print("FocusHasIronBubblegum: " + str(FocusHasIronBubblegum))
    print("FocusHasHeliumBubblegum: " + str(FocusHasHeliumBubblegum))
    print("FocusHasShell: " + str(FocusHasShell))
    print("FocusHasTrombone: " + str(FocusHasTrombone))

    print("UnusedExtraSense30: " + str(UnusedExtraSense30))
    print("UnusedExtraSense31: " + str(UnusedExtraSense31))
    print("")

    #return entity