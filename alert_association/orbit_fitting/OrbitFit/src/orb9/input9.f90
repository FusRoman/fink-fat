! **********************************************************            
!  {\bf input8v} vers. 2.1                                              
!  options, initial conditions, filters                                 
!                                                                       
!  initial conditions are converted to coordinates specified by         
!  cooy =CAR and sysy = BAR                                             
! **********************************************************            
SUBROUTINE inpo9(dt,nout,idump,iprqua,                      &
     &             sysz,refz,refy,                                      &
     &             v0,v1,vpos,vvel,vslog,                               &
     &             t1,yp,ya,ngip,ngia,yv)                               
  USE fund_const
  USE massmod
  USE fir_filters
  USE force9d
  USE force_model, ONLY: iclap
  USE close_app, ONLY: npoint
  USE planet_masses, ONLY: dmin
  USE output_control, ONLY: iuncla
  USE propag_state, ONLY: intein
  USE dyn_param, ONLY: iyark
  USE force_model, ONLY: velo_req
  IMPLICIT NONE 
! commons                                                               
  INCLUDE 'comnbo.h90' 
! =======================OUTPUT========================                 
! options: output interval                                              
  DOUBLE PRECISION, INTENT(OUT)  :: dt 
! options: number outputs, dump interval, output control 
  INTEGER, INTENT(OUT)  :: nout,idump,iprqua 
!  identifiers of coordinate systems                                    
  character*3, INTENT(OUT)  :: sysz 
  character*6, INTENT(OUT)  :: refy,refz 
! filter coefficients                                                   
  LOGICAL found 
! normalisation controls                                                
  DOUBLE PRECISION, INTENT(OUT)  :: v0,v1,vvel,vpos,vslog(nvzx) 
! arrays for initial conditions: epoch, planets, asteroids              
  DOUBLE PRECISION, INTENT(OUT)  :: t1, yp(6,nbox), ya(6,nastx) 
! revolution counters                                                   
  INTEGER, INTENT(OUT)  :: ngia(nastx),ngip(nbox) 
! variations                                                            
  DOUBLE PRECISION, INTENT(OUT)  :: yv(6,nvzx) 
! ====================END INTERFACE====================                 
! arrays for initial conditions                                         
  DOUBLE PRECISION xp(6,nbox),bar(6) 
  DOUBLE PRECISION xa(6,nastx) 
  INTEGER nga(nangx,nastx),ngp(nangx,nbox) 
  DOUBLE PRECISION barin(6) 
! to compute step, elements in output coordinates                       
  DOUBLE PRECISION enne(norbx),a(norbx),ecc(norbx) 
!  identifiers of coordinate systems                                    
  character*3 coox,sysx,cooz 
  character*6 refx,refp 
!  names and numbers of input planets and asteroids, comments           
  character*4 number(nastx) 
  INTEGER nfound 
!  flag for catalog input/input from output; existence of filters       
  logical catalo 
! close aproach control                                                 
  DOUBLE PRECISION dmint,dminj 
! arrays for variational equations                                      
  DOUBLE PRECISION xv(6,nvzx),jjin(nastx) 
  DOUBLE PRECISION semim 
! epoch and initial times                                               
  DOUBLE PRECISION t0p,dtin,t0,t0a,tp,t0b 
! dimensions, controls                                                  
  INTEGER norb,nast,ibar,npla,nang,iun,ilca 
! loop indexes                                                          
  INTEGER i,j , jj, k, le
! functions                                                             
  INTEGER numang 
! asteroid names for yarkovsky effect                                   
  CHARACTER*9 idac(nastx) 
  DOUBLE PRECISION dadtin(nastx)
  LOGICAL isya
! **********************************************************            
!  open option file and output file, headers                            
  open(1,file='orb9.opt',status='old') 
  open(9,file='orb9.out',status='unknown') 
  write(9,110) 
110 format('; orbit9 vers. c, output file') 
!  input options                                                        
  CALL reaopt(1,ibar,catalo,dt,nout,idump,nsamp,iprqua,sysz,refz,v1,semim)
! select right hand side no. 3
  rhs=3
! **********************************************************            
!  barycenter correction, if required, from unit 7                      
! **********************************************************            
!  planetary initial conditions, from unit 2:                           
! **********************************************************            
  CALL reapla(2,7,9,ibar,iprqua,                                    &
     &    coox,sysx,refp,sysz,refz,nbod,t0p,tp,xp,ngp,barin)            
  npla=nbod-1 
  refx=refp 
! *******************************************************               
!  reduction to standard reference system, computation of no.rev        
! *******************************************************               
  refy=refz 
  CALL coostp(coox,sysx,refx,nbod,xp,ngp,                           &
     &        sysz,refz,ibar,barin,yp,ngip,bar,a,ecc,enne)              
! fix 17/12/2008 A.Milani: reset to zero number of revs,
! to compensate for sum of three angles which may be more than one rev
  ngip=0
! **********************************************************            
!  asteroids initial conditions, from unit 3:                           
! **********************************************************            
! list of required objects                                              
  write(9,422)na,nvz,(ida(j),j=1,na) 
422 format(i4,'  asteroids required, ',i3,'  with LCE; numbers:'/(8(2x,a9)))
! input: two cases                                                      
  if(catalo)then 
! catalog of asteroid orbits                                            
     CALL reacat(iprqua,sysz,refz,coox,sysx,refx,t0a,t1,xa,nfound)
! limitations of present catalogues                                     
!                                                                       
! no number of revolutions                                              
     nang=numang(coox) 
     DO j=1,na 
        DO i=1,nang 
           nga(i,j)=0 
        ENDDO
     ENDDO
!  (6a) starting value for divergence ratio;                            
!  no variational equation appear in catalogues (so far)                
     ilca=0 
     do j=1,nvz 
        vslog(j)=0.d0 
        jjin(j)=1 
     enddo
  else 
!  input from output of a previous integration                          
     CALL reaout(iprqua,sysz,refz,coox,sysx,refx,t0a,t1             &
     &      ,xa,nga,vslog,jjin,xv,nfound)                               
     IF(nfound.ne.na)THEN 
        WRITE(*,*)' not found in previous output, missing;' 
        WRITE(*,*)(ida(j),j=nfound+1,na) 
        STOP 
     ENDIF
  endif
!  (2) check consistency of reference times                             
  if(abs(t0a-t0p).gt.1.d-7)then 
     write(9,*)' reference times incompatible',t0a,t0p 
     stop 
  else 
     t0=t0p 
  endif
!  (3a) check consistency of input time: all at the same epoch          
  if(abs(tp-t1).gt.1.d-7)then 
     write(9,*)' initial times incompatible',t1,tp 
     stop 
  else 
     t1=tp 
  endif
!  if barycenter correction is required, it must all be consistent      
  if(ibar.eq.1)then 
     if(refx.ne.refp)then 
        write(9,*)' reference system of baryc. inconsistent' 
        write(9,*)' with asteroid input ',refp,refx 
        stop 
     endif
  endif
!                                                                       
  close(3) 
! *******************************************************               
! asteroids:                                                            
!  reduction to standard reference system, computation of no.rev        
! *******************************************************               
  CALL coosta(coox,sysx,refx,na,xa,nga,sysz,refz,nbod,              &
     & bar,ibar,barin,ya,ngia,a(nbod),ecc(nbod),enne(nbod))  
! fix 17/12/2008 A.Milani: reset to zero number of revs,
! to compensate for sum of three angles which may be more than one rev
  ngia=0
! *******************************************************               
! variational equations initial conditions                              
! *******************************************************               
  CALL vareqi(ilca,semim,jjin,xv,v0,v1,vvel,vpos,vslog,yv) 
! ********************************************************************  
!  options and initialisations for the propagator                       
!  the elements are taken from the output system za, zp,                
!  which is always in equinoctal elements                               
! ********************************************************************  
  norb=na+npla 
!  dtin is the filter input time interval (to adjust stepsize to        
!  an integer submultiple                                               
  dtin=dt/nsamp 
  CALL intein(1,dtin,a,enne,ecc,norb,norbx)
  
! WARNING: need selste_orbit9 to select stepsize, replacing intein 
! ********************************************************************  
!  input controls for close approach monitoring                         
! ********************************************************************  
  call skip(1,1) 
  call reaflo(1,'dmint',dmint) 
  call reaflo(1,'dminj',dminj) 
  call reaint(1,'npoint',npoint) 
!  which is a giant planet?                                             
  DO 52 j=1,npla 
     if(gm(j+1)/gm(1).gt.1d-5)then 
        dmin(j)=dminj**2 
     else 
        dmin(j)=dmint**2 
     endif
52 ENDDO
! however, to slow down the integration to monitor a close              
! approach is not possible, because of parallel propagation of many orbi
  iclap=1
! but this is done in a way which is special to orbit9, controlled by rhs=3
!   close approach file: WARNING: iuncla is in output_control
  iuncla=66 
  open(iuncla,file='orb9.clo',status='unknown') 
!  header would be needed...                                            
!                                                                       
! other orbfit style controls for optional perturbations              
  call skip(1,1)
  call reaint(1,'irelj2',irelj2)
  call reaint(1,'iyark', iyark)
  close(1) ! end options
! input of Yarkovsky data                                        
! note: the file yarkosky.dat contains the value of dadt in au/My  
  IF(iyark.eq.3)THEN   
     INQUIRE (file ='yarkovsky.dat', exist = isya)
     IF(.not.isya)THEN
        STOP '*** missing yarkovsky.dat*******'
     ELSE
        velo_req=.true. ! propagator need to provide velocity at each step
        OPEN(99,file='yarkovsky.dat',status='unknown') 
        jj=0 
        DO j=1,nastx 
           READ(99,*,end=5) idac(j),dadtin(j) 
           CALL rmsp(idac(j),le) 
           WRITE(*,*) idac(j),dadtin(j) 
           jj=jj+1 
        ENDDO
5       CONTINUE 
        DO j=1,na 
           dadt9(j)=0.d0 
           DO k=1,jj 
              IF(idac(k).eq.ida(j))THEN 
                 dadt9(j)=dadtin(k) 
              ENDIF
           ENDDO
        ENDDO
        CLOSE(99)
     ENDIF
  ELSE
     velo_req=.false.
  ENDIF
! ********************************************************************  
!  input filter coefficients                                            
! ********************************************************************  
  CALL filope(nsamp,1)
! *******************************************************************   
  write(9,*)' input complete, starting from t=',t1 
! *************************************************************         
!  all the input error cases                                            
                                                                        
END SUBROUTINE inpo9
