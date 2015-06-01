PROGRAM RDF_calculation

! To calculate the RDF g(r) radial distribution function between two atom sites
! Exclude the intramolecular RDF
! Read lammps dump file, format (id type xu yu zu) ! make sure the coords are unwrapped!
! Written Feb 18 2010
! Hao Wu, Univ. Notre Dame

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Changes made by Matthew Wade
! to modify program for CED post processing
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Last change made 6/24/14
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Added section that reads pair, bond, angle, dihedral, and improper coeffients
!Added section to read ctrl from universal control file

IMPLICIT NONE

INTEGER :: i,k,kk,b,p,g,m,rr,mm,ii,j,nn, nspecies, nrdfs, ndump_pts
INTEGER :: time_step,line_n,pro_point,atom_n,freq,atom,id,atom_type,site,mol_id
INTEGER :: bin_n,ch,atom_id
INTEGER :: site1_n,site2_n,ideal_pair,which_bin,mol_id1,mol_id2
INTEGER :: start_atom, numa1, numa2, numas1, numas2
INTEGER :: i_timestep, natoms_box, dummy1, dummy2, atomnum, atomtemp
INTEGER :: iatom1, iatom2, site1_s, site2_s, num_bins
INTEGER :: timestart, timestop
INTEGER :: counter
INTEGER :: nbonds_tot, natoms_tot, nangles_tot, ndih_tot, nimp_tot
INTEGER :: batom1, batom2
INTEGER :: aatom1, aatom2, aatom3
INTEGER :: datom1, datom2, datom3, datom4
INTEGER :: nsnapshots
INTEGER :: nbr_entries, array_error,ierr,line_nbr
INTEGER :: dummyint
INTEGER :: nskip

INTEGER,ALLOCATABLE :: site1_a(:), site2_a(:)
INTEGER,ALLOCATABLE :: bondedto(:,:), numbondedto(:)
INTEGER,ALLOCATABLE :: angledto(:,:), numangledto(:)
INTEGER,ALLOCATABLE :: dihto(:,:), numdihto(:)
INTEGER,ALLOCATABLE :: natoms_s(:), nmols_s(:)
INTEGER,ALLOCATABLE :: atom1(:), atom2(:)
LOGICAL             :: bondedpair, priorfound

! line_n is the number of lines per timestep in dump file
! atom_n    is the number of atoms per replicate unit
! freq      is the frequency of dump file written
! pro_point is the number of production data points

REAL :: box_x,box_y,box_z,xlo,xhi,ylo,yhi,zlo,zhi,x,y,z
REAL :: x1,y1,z1,x2,y2,z2,dx,dy,dz,r,density,volume,shell,distance,r_bin
REAL :: dummyreal
REAL,parameter :: pi=3.141593
REAL,allocatable :: gr(:)
REAL :: bin_wid, rcut
REAL,allocatable :: xp(:), yp(:), zp(:)
CHARACTER(120) :: dummy,dumpfile,ldumpfile,filename,extension,datafile
CHARACTER(120) :: line_string, line_array(20), err_msg
CHARACTER(100) :: string
CHARACTER(10)  :: element

!Variables added by Matthew Wade to adjust for CED post processing
INTEGER :: timestep
INTEGER :: typ
INTEGER :: natoms_typ, nbonds_typ, nangles_typ, ndih_typ, nimp_typ
INTEGER, ALLOCATABLE :: bondtype(:,:), angletype(:,:), dihtype(:,:), imptype(:,:)
REAL, ALLOCATABLE :: pair_coeff(:,:), bond_coeff(:,:), angle_coeff(:,:)
REAL, ALLOCATABLE :: dih_coeff(:,:), imp_coeff(:,:)
REAL :: vdwlscale, coulscale, q, vdwl_energy, coul_energy
INTEGER :: molid, aid 
INTEGER, ALLOCATABLE :: atomtype(:), molnum(:)
REAL, ALLOCATABLE :: charge(:), vdwl_total(:), coul_total(:)
REAL :: ep, sig, eptmp, sigtmp
REAL :: prefac1, prefac2, molecules
LOGICAL :: doexist, flag 

REAL :: vtot, ctot

!For use in writing of trajectory file
INTEGER :: atomid, atype
REAL :: xpos, ypos, zpos, atomcharge

vtot = 0
ctot = 0

flag = .true.
molecules = 110 !Number ofmolecules in the system, so far only used when calculating prefacs
!prefac2 = 5.510e-22 / (1/6.022e23) !this is the constant used to swap units when calculating coulomb energy
prefac2 = 332.06371
prefac1 = 1.0                        !this is the constant used to swap units when calculating van der waals energy 

!OPEN(UNIT=11,FILE='postproc.ctrl',STATUS='OLD',ACTION='READ')
OPEN(UNIT=10,FILE='CAL.ctrl',STATUS='OLD',ACTION='READ')

! read the control file
READ(10,*)
READ(10,*) molecules
READ(10,*) 
READ(10,*) numa1
READ(10,*) 
READ(10,*) !molecular weight -> do not need in this program
READ(10,*) 
READ(10,*) datafile
READ(10,*) 
READ(10,*) !in file -> do not need in this program
READ(10,*) 
READ(10,*) ldumpfile
READ(10,*) 
READ(10,*) nsnapshots

numa2 = numa1
ldumpfile = TRIM(ldumpfile)
nskip = 0


! Now we get bonded information.  We can make this optional later

!CLOSE(11)
OPEN(UNIT=12,FILE=datafile,STATUS='OLD',ACTION='READ')

READ(12,*)
READ(12,*)
READ(12,*) natoms_tot
READ(12,*) nbonds_tot
READ(12,*) nangles_tot
READ(12,*) ndih_tot
READ(12,*) nimp_tot

ALLOCATE(xp(natoms_tot),yp(natoms_tot),zp(natoms_tot))
xp(:) = 0.0
yp(:) = 0.0
zp(:) = 0.0

READ(12,*)
READ(12,*) natoms_typ
READ(12,*) nbonds_typ
READ(12,*) nangles_typ
READ(12,*) ndih_typ
READ(12,*) nimp_typ

ALLOCATE(pair_coeff(natoms_typ, 2), bond_coeff(nbonds_typ, 2), angle_coeff(nangles_typ, 2), &
         dih_coeff(ndih_typ, 4), imp_coeff(nimp_typ, 2))
pair_coeff(:,:) = 0.0
bond_coeff(:,:) = 0.0
angle_coeff(:,:) = 0.0
dih_coeff(:,:) = 0.0
imp_coeff(:,:) = 0.0


IF(nbonds_tot .GT. 0) THEN
   ALLOCATE(bondedto(4,natoms_tot))
   ALLOCATE(bondtype(4,natoms_tot))
   ALLOCATE(numbondedto(natoms_tot))

   bondedto(:,:) = 0
   bondtype(:,:) = 0
   numbondedto(:) = 0

   DO

      READ(12,'(A)') string
      IF(string(1:5) == 'Bonds') EXIT

   END DO

   READ(12,*) ! Blank line

   DO i = 1, nbonds_tot

      READ(12,*) dummy1, typ, batom1, batom2
      numbondedto(batom1) = numbondedto(batom1) + 1
      numbondedto(batom2) = numbondedto(batom2) + 1
      bondedto(numbondedto(batom1),batom1) = batom2
      bondedto(numbondedto(batom2),batom2) = batom1
      bondtype(numbondedto(batom1),batom1) = typ
      bondtype(numbondedto(batom2),batom2) = typ

   END DO
END IF

IF(nangles_tot .GT. 0) THEN
   ALLOCATE(angledto(12,natoms_tot))
   ALLOCATE(angletype(12, natoms_tot))
   ALLOCATE(numangledto(natoms_tot))

   angledto(:,:) = 0
   angletype(:,:) = 0
   numangledto(:) = 0

   DO

      READ(12,'(A)') string
      IF(string(1:6) == 'Angles') EXIT

   END DO

   READ(12,*) ! Blank line

   DO i = 1, nangles_tot

      READ(12,*) dummy1, typ, aatom1, aatom2, aatom3
      numangledto(aatom1) = numangledto(aatom1) + 1
      numangledto(aatom3) = numangledto(aatom3) + 1
      angledto(numangledto(aatom1),aatom1) = aatom3
      angledto(numangledto(aatom3),aatom3) = aatom1
      angletype(numangledto(aatom1), aatom1) = typ
      angletype(numangledto(aatom3), aatom3) = typ

   END DO
END IF

IF(ndih_tot .GT. 0) THEN
   ALLOCATE(dihto(12,natoms_tot))
   ALLOCATE(dihtype(12, natoms_tot))
   ALLOCATE(numdihto(natoms_tot))

   dihto(:,:) = 0
   dihtype(:,:) = 0
   numdihto(:) = 0

   DO

      READ(12,'(A)') string
      IF(string(1:9) == 'Dihedrals') EXIT

   END DO

   READ(12,*) ! Blank line

   DO i = 1, ndih_tot

      READ(12,*) dummy1, typ, datom1, datom2, datom3, datom4
      numdihto(datom1) = numdihto(datom1) + 1
      numdihto(datom4) = numdihto(datom4) + 1
      dihto(numdihto(datom1),datom1) = datom4
      dihto(numdihto(datom4),datom4) = datom1
      dihtype(numdihto(datom1),datom1) = typ
      dihtype(numdihto(datom4),datom4) = typ

   END DO
END IF


!Read all the types into the program
DO

    READ(12,'(A)') string
        IF(string(1:12) == 'Pair Coeffs') EXIT

END DO

IF(natoms_typ .GT. 0) THEN
   READ(12,*) !blank line

   DO i = 1, natoms_typ
      READ(12,*) dummy1, eptmp, sigtmp

      pair_coeff(dummy1, 1) = eptmp
      pair_coeff(dummy1, 2) = sigtmp

      !WRITE(*,*) dummy1, eptmp, sigtmp

   END DO
END IF

IF(nbonds_typ .GT. 0) THEN
   READ(12,*)
   READ(12,*)
   READ(12,*)

   DO i = 1, nbonds_typ
      READ(12,*) !dummy1, bond_coeff(i, 1), bond_coeff(i, 2)
   END DO

END IF

IF(nangles_typ .GT. 0) THEN
   READ(12,*)
   READ(12,*)
   READ(12,*)

   DO i = 1, nangles_typ
      READ(12,*) !dummy1, angle_coeff(i, 1), angle_coeff(i, 2)
   END DO

END IF

IF(ndih_typ .GT. 0) THEN
   READ(12,*)
   READ(12,*)
   READ(12,*)

   DO i = 1, ndih_typ
      READ(12,*) !dummy1, dih_coeff(i, 1), dih_coeff(i, 2), dih_coeff(i, 3), dih_coeff(i, 4)
   END DO

END IF

!Impropers not considered for vdwl and coul energy calculations
!IF(nimp_typ .GT. 0) THEN
!   READ(12,*)
!   READ(12,*)
!   READ(12,*)
!
!   DO i = 1, nimp_typ
!      READ(12,*) dummy1, imp_coeff(i, 1), imp_coeff(i, 2)
!   END DO
!
!END IF

! We have all bonded information


! We can now read the dump file and get the xyz coordinates for the atoms we
! just figured out are important.

OPEN(60,file=ldumpfile)    !dump file set to ch 60
   
ALLOCATE(atomtype(natoms_tot), molnum(natoms_tot))
ALLOCATE(charge(natoms_tot))
ALLOCATE(vdwl_total(molecules), coul_total(molecules))     
DO i = 1, nsnapshots
   READ(60,*,IOSTAT=ierr) line_string
   READ(60,*,IOSTAT=ierr) timestep
   !WRITE(*,*) line_string
   WRITE(*,*) timestep
   DO j = 1, 3 !read over unimportant lines return error if numbers hit
      READ(60,'(A)',IOSTAT=ierr) line_string
      !WRITE(*,*) line_string
   END DO

   CALL Parse_String(60,line_nbr,2,nbr_entries,line_array,ierr)
   xlo = String_To_Double(line_array(1)) 
   xhi = String_To_Double(line_array(2)) 
   CALL Parse_String(60,line_nbr,2,nbr_entries,line_array,ierr)
   ylo = String_To_Double(line_array(1)) 
   yhi = String_To_Double(line_array(2)) 
   CALL Parse_String(60,line_nbr,2,nbr_entries,line_array,ierr)
   zlo = String_To_Double(line_array(1)) 
   zhi = String_To_Double(line_array(2)) 

   box_x = xhi - xlo
   box_y = yhi - ylo
   box_z = zhi - zlo
   volume = box_x*box_y*box_z !got volume of box

   atomtype(:) = 0
   molnum(:) = 0
   charge(:) = 0
   vdwl_total(:) = 0
   coul_total(:) = 0

   READ(60,'(A)',IOSTAT=ierr) line_string
   

   DO ii = 1, natoms_tot

      !WRITE(*,*) ii
!      CALL Parse_String(60,line_nbr,4,nbr_entries,line_array,ierr)
      READ(60,*) id, molid, aid, x, y, z, q
!      WRITE(*,*) id, molid, aid, x, y, z, q, ii
!      id = String_To_Double(line_array(1))
!      x = String_To_Double(line_array(nbr_entries-2))
!      y = String_To_Double(line_array(nbr_entries-1))
!      z = String_To_Double(line_array(nbr_entries))
      !WRITE(*,*) id
      xp(id) = x
      yp(id) = y
      zp(id) = z

      atomtype(id) = aid        !there has to be a better way to do that
      molnum(id) = molid  
      charge(id) = q


   END DO

   !IF(i .LE. nskip) CYCLE !for my experiment i dont think that we will need this as all
                          !data files can be used 

   ! We now have all the positions of the atoms for this snapshot.
   ! We then calculate this snapshot's contribution to the rdf

!   IF(site1_s == site2_s) THEN
!      ideal_pair = numas1*(numas2 - 1)/2 ! If the atoms are on the same species then
!   ELSE                                  ! the bulk density is given by the first eq.
!      ideal_pair = numas1*numas2
!   END IF                                ! If not on the same species the the second eq.
!  
!   density = REAL(ideal_pair)/volume\

   DO j = 1,numa1 ! We now have a double loop 

      iatom1 = j
      x1 = xp(iatom1)
      y1 = yp(iatom1)
      z1 = zp(iatom1)

      DO k = 1,numa2

         iatom2 = k
         IF (iatom2 .LE. iatom1) CYCLE
         IF(.NOT.(molnum(iatom1) == molnum(iatom2))) CYCLE !ignore all atoms except those in one molecule

         bondedpair = .FALSE.
         priorfound = .FALSE.
         vdwlscale = 1.0
         coulscale = 1.0
         IF(nbonds_tot .GT. 0) THEN
            DO kk = 1,numbondedto(iatom1)
               IF(bondedto(kk,iatom1) == iatom2) THEN 
                  bondedpair = .TRUE.
                  priorfound = .TRUE.
                  vdwlscale = 0.0
                  coulscale = 0.0
               END IF
            END DO
         END IF
         IF(nangles_tot .GT. 0) THEN
            DO kk = 1,numangledto(iatom1)
               IF(angledto(kk,iatom1) == iatom2) THEN
                  bondedpair = .TRUE.
                  priorfound = .TRUE.
                  vdwlscale = 0.0
                  coulscale = 0.0
               END IF
            END DO
         END IF
         IF(ndih_tot .GT. 0 .AND. priorfound .EQV. .FALSE.) THEN
            DO kk = 1,numdihto(iatom1)
               IF(dihto(kk,iatom1) == iatom2) THEN
                  bondedpair = .FALSE.
                  vdwlscale = 0.5
                  coulscale = 0.5
               END IF
            END DO
         END IF

         IF(bondedpair) CYCLE !Loop back to new atom, we are going to disregard anything 

         x2 = xp(iatom2)
         y2 = yp(iatom2)
         z2 = zp(iatom2)

         dx=abs(x1-x2)
         dy=abs(y1-y2)
         dz=abs(z1-z2)

         !dx=dx-box_x*ANINT(dx/box_x) !wraps distances into box?
         !dy=dy-box_y*ANINT(dy/box_y)
         !dz=dz-box_z*ANINT(dz/box_z)
         r=SQRT(dx*dx+dy*dy+dz*dz)  ! Distance between the two atoms.  Don't have to 
                                    ! apply periodic boundary conditions because the
                                    ! positions are unwrapped.

         dummyint = molnum(iatom1)

         ep = SQRT(pair_coeff(atomtype(iatom1),1) * pair_coeff(atomtype(iatom2), 1))
         sig = (pair_coeff(atomtype(iatom1),2) + pair_coeff(atomtype(iatom2), 2))/2
         
         !ep = sqrt(pair_coeff(atomtype(iatom1),2) * pair_coeff(atomtype(iatom2), 2))
         !sig = (pair_coeff(atomtype(iatom1),1) + pair_coeff(atomtype(iatom2), 1))/2

         !WRITE(*,*) "scaling factors (vdwl, coul)"
         !WRITE(*,*) vdwlscale
         !WRITE(*,*) coulscale

         vdwl_energy = prefac1 * 4 * ep * ((sig/r)**12 - (sig/r)**6)
         vdwl_total(dummyint) = vdwl_total(dummyint) + vdwlscale * vdwl_energy
        
         coul_energy = prefac2 * charge(iatom1) * charge(iatom2) / r
         coul_total(dummyint) = coul_total(dummyint) + coulscale * coul_energy 
         !WRITE(*,*)dummyint, iatom1, iatom2, atomtype(iatom1), atomtype(iatom2), vdwl_energy, coul_energy, r
         !WRITE(*,*)dummyint, iatom1, charge(iatom1), iatom2, charge(iatom2), coul_energy, r
         !WRITE(*,*) dummyint, iatom1, pair_coeff(atomtype(iatom1),1), pair_coeff(atomtype(iatom1),2)
         !WRITE(*,*) dummyint, iatom2, pair_coeff(atomtype(iatom2),1), pair_coeff(atomtype(iatom2),2)
      END DO
   END DO
   
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !!!!!!!!!!!!!Output "dump file" for MAPS to read!!!!!!!!!!!!!!!!!!!!
   !!!!!!!!!!!!!!Only doing this for molecule 1   !!!!!!!!!!!!!!!!!!!!!
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   IF (i == 1) THEN    !get data from first snapshot b/c everything is a running total
      OPEN(85, file='firstmolec.dump', IOSTAT=ierr, action="write")
      WRITE(85,*) "ITEM: TIMESTEP"
      WRITE(85,*) timestep
      WRITE(85,*) "ITEM: NUMBER OF ATOMS"
      WRITE(85,*) natoms_tot
      WRITE(85,*) "ITEM: BOX BOUNDS p p p"
      WRITE(85,*) xlo, xhi
      WRITE(85,*) ylo, yhi
      WRITE(85,*) zlo, zhi
      WRITE(85,*) "ITEM: ATOMS id mol type xu yu zu q"
     
      DO atomid = 1, natoms_tot 
         IF (molnum(atomid) == 1) THEN !only get info for first molecule
            atype = atomtype(atomid) 
            xpos = xp(atomid)
            ypos = yp(atomid)
            zpos = zp(atomid)
            atomcharge = charge(atomid)
            WRITE(85,*) atomid, molnum(atomid), atype, xpos, ypos, zpos, atomcharge
         END IF
      END DO
      CLOSE(85)
      OPEN(80, file='firstmolec.energy', IOSTAT=ierr, action="write")
      WRITE(80,*) "VDWL     COUL  test"
      WRITE(80,*) vdwl_total(1), coul_total(1)
      CLOSE(80)
   END IF

   IF (flag) THEN
       OPEN(90,file='intmolec.dat',IOSTAT=ierr) !replace old file
       flag = .FALSE.
   ELSE
       OPEN(90,file='intmolec.dat', STATUS="old", POSITION="append", action="write") !append to file that exists
   END IF
   
 
   WRITE(90,*) "TIMESTEP"
   WRITE(90,*) timestep
   DO p=1,molecules
      WRITE(90,*)p, vdwl_total(p), coul_total(p)
      vtot = vtot + vdwl_total(p)
      ctot = ctot + coul_total(p)
   END DO
   CLOSE(90)

   OPEN(91, file='totals.foo', IOSTAT=ierr)
   WRITE(91,*) "vtot ctot"
   WRITE(91,*) vtot, ctot

   CLOSE(91)

WRITE(*,*) "Done set"

END DO !Loop through new snapshot

CLOSE(12)
CLOSE(60)

! write the results

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

CONTAINS

  SUBROUTINE Read_String(file_number,string,ierr)
!********************************************************************************
! This routine just reads a single string from a file with unit number
! equal to file_number and returns that 120 character string. It returns
! ierr .ne. 0 if it couldn't read the file.
!********************************************************************************
    INTEGER, INTENT(IN) :: file_number
    INTEGER, INTENT(OUT) :: ierr
    CHARACTER(120), INTENT(OUT) :: string
!********************************************************************************

    READ(file_number,'(A120)',IOSTAT=ierr) string
    IF (ierr .NE. 0) RETURN


  END SUBROUTINE Read_String

!********************************************************************************
  SUBROUTINE Parse_String(file_number,line_nbr,min_entries,nbr_entries,line_array,ierr) 
!********************************************************************************
! This routine reads one line from the file file_number. It reads the total number
! of entries on the line and places the entries in the character array line_array
! in consecutive order. It skips leading blanks, and determines if an entry is 
! different by detecting a space between entries. It also tests to see if the 
! minimum number of entries specified was met or not. If not, and error is returned.
!********************************************************************************
    CHARACTER(120), INTENT(OUT) :: line_array(20)
    INTEGER, INTENT(IN) :: file_number,min_entries,line_nbr
    INTEGER, INTENT(OUT) :: nbr_entries
    INTEGER, INTENT(INOUT) :: ierr
    
    CHARACTER(700) :: string
    INTEGER :: line_position,i
    LOGICAL :: space_start
!********************************************************************************

! Zero counter for number of entries found
    nbr_entries = 0

! Counter for positon on the line
    line_position = 1

! clear entry array
    line_array = ""
      
! Read the string from the file
    CALL Read_String(file_number,string,ierr)
    
    IF (string(1:1) .NE. ' ') THEN

       ! first character is an entry, so advance counter
       nbr_entries = nbr_entries + 1
    ENDIF

    space_start = .FALSE. 

    ! Recall: LEN_TRIM() is length of string, not counting trailing blanks
    DO i=1,LEN_TRIM(string) 
       IF (string(i:i) .EQ. ' ') THEN
          IF (.NOT. space_start) space_start = .TRUE.
          ! This means a new set of spaces has been found
       ELSE
          IF (space_start) THEN
             nbr_entries = nbr_entries + 1
             line_position = 1
          ENDIF
          space_start = .FALSE.
          line_array(nbr_entries)(line_position:line_position) = &
               string(i:i)
          line_position = line_position + 1
       ENDIF
       
    ENDDO

    ! Test to see if the minimum number of entries was read in      
    IF (nbr_entries < min_entries) THEN
       err_msg = ""
       err_msg = 'Expected at least '// TRIM(Int_To_String(min_entries))//&
            ' input(s) on line '//TRIM(Int_To_String(line_nbr))//' of input file.'
       WRITE(*,'(A)') err_msg
       STOP
    END IF
      
    END SUBROUTINE Parse_String

!****************************************************************************

!****************************************************************************
FUNCTION Int_To_String(int_in)
!****************************************************************************
      IMPLICIT NONE
  !**************************************************************************
  !                                                                         *
  ! This function takes an integer argument
  ! and returns the character equivalent
  !                                                                         *
  !**************************************************************************
  CHARACTER(40) :: int_to_string
  LOGICAL :: is_negative
  INTEGER :: int_in, chop_int, ndigits, curr_digit
  
  int_to_string = ""
  ndigits = 0
  
  !Check to see if integer is zero
  IF (int_in == 0) THEN
     int_to_string(1:1) = "0"
     RETURN
  END IF
  
  !Determine if integer is negative
  IF (int_in < 0) THEN
     is_negative = .TRUE.
     chop_int = -int_in
  ELSE
     is_negative = .FALSE.
     chop_int = int_in
  END IF
  
  !Pop of last digit of integer and fill in string from right to
  !left
  DO
     ndigits = ndigits + 1
     curr_digit = MOD(chop_int,10)
     int_to_string(41-ndigits:41-ndigits) = ACHAR(curr_digit+48)
     chop_int = INT(chop_int / 10.0d0)
     IF (chop_int == 0) EXIT
  END DO
  
  IF (is_negative) int_to_string(40-ndigits:40-ndigits) = "-"
  
  !Left justify string
  int_to_string = ADJUSTL(int_to_string)

END FUNCTION Int_To_String

FUNCTION String_To_Double(string_in)
!****************************************************************************
! This function takes a character string in and outputs the equivalent DP 
  ! number.
!****************************************************************************

  LOGICAL :: dec_found, exp_found, is_negative
  CHARACTER(*) :: string_in
  CHARACTER(50) :: cff_bd, cff_ad, expv
  INTEGER :: nchars, strln, exp_start,dec_start
  INTEGER :: ii, cnt, icff_bd, iexpv
  REAL :: string_to_double, div, add_num
!****************************************************************************
  !Initialize some things
  string_to_double = 0.0
  dec_found = .FALSE.
  exp_found = .FALSE.
  cff_bd = ""
  cff_ad = ""
  expv = ""
  string_in = ADJUSTL(string_in)
  nchars = LEN_TRIM(string_in)
  strln = LEN(string_in)
  is_negative = .FALSE.
  IF (string_in(1:1) == "-") is_negative = .TRUE.
  
  ! Make an initial pass through the string to find
  ! if and where the decimal and exponent marker are
  exp_start = -1
  dec_start = -1
  DO ii = 1, nchars
     IF (string_in(ii:ii) == ".") dec_start = ii
     IF (string_in(ii:ii) == "D") exp_start = ii
     IF (string_in(ii:ii) == "d") exp_start = ii
     IF (string_in(ii:ii) == "E") exp_start = ii
     IF (string_in(ii:ii) == "e") exp_start = ii
  END DO
  IF (exp_start > 0) THEN
     exp_found = .TRUE.
  ELSE
     exp_start = nchars + 1
  END IF
  IF (dec_start > 0) THEN
     dec_found = .TRUE.
  ELSE
     dec_start = exp_start
  END IF

  !Based on above, break string into components
  cnt = 1
  DO ii = 1, dec_start - 1
     cff_bd(cnt:cnt) = string_in(ii:ii)
     cnt = cnt + 1
  END DO
  
  IF (dec_found) THEN
     cnt = 1
     DO ii = dec_start + 1, exp_start - 1
        cff_ad(cnt:cnt) = string_in(ii:ii)
        cnt = cnt + 1
     END DO
  ELSE
     cff_ad(1:1) = "0"
  END IF
       
  IF (exp_found) THEN
     cnt = 1
     DO ii = exp_start + 1, nchars
        expv(cnt:cnt) = string_in(ii:ii)
        cnt = cnt + 1
     END DO
  ELSE
     expv(1:1) = "0"
  END IF
  
  !Convert exponent, predecimal components to integers
  icff_bd = string_to_int(cff_bd)
  iexpv= string_to_int(expv)

  !Combine components into real number
  string_to_double = REAL(icff_bd)
  div = 10.0
  DO ii = 1, LEN_TRIM(cff_ad)
     add_num = (IACHAR(cff_ad(ii:ii)) - 48) / div
     IF (is_negative) add_num = -add_num
     string_to_double = string_to_double + add_num
     div = div*10.0
  END DO
  string_to_double = string_to_double*10.0**iexpv

END FUNCTION String_To_Double


!****************************************************************************
FUNCTION String_To_Int(string_in)
!****************************************************************************
  ! This function takes a character string as input out returns the 
  ! equivalent integer.
!****************************************************************************

  LOGICAL :: is_negative
  INTEGER :: string_to_int, ndigits, strln
  INTEGER :: mult, digit, pos, ii
  CHARACTER(*) :: string_in
!****************************************************************************
  !Initialize some things
  string_to_int = 0
  string_in = ADJUSTL(string_in)
  ndigits = LEN_TRIM(string_in)
  strln = LEN(string_in)

  !Find out if the number is negative
  is_negative = .FALSE.
  IF (string_in(1:1) == "-") THEN
     is_negative = .TRUE.
     ndigits = ndigits - 1
  END IF
  IF (string_in(1:1) == "+") ndigits = ndigits - 1

  !Pull of digits starting at the end, multiply by
  !the correct power of ten and add to value
  string_in = ADJUSTR(string_in)
  mult = 1
  DO ii = 1, ndigits
     pos = strln - ii + 1
     digit = IACHAR(string_in(pos:pos)) - 48
     string_to_int = string_to_int + mult*digit
     mult = mult*10
  END DO
     
  !If it's negative, make it so
  IF (is_negative) string_to_int = -string_to_int

END FUNCTION String_To_Int

END PROGRAM
