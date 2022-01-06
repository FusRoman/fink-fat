MODULE force9d
  USE massmod
  IMPLICIT NONE 
  PRIVATE

! shared data

INTEGER, PUBLIC :: irelj2 ! flag for J2 and relativistic perturbations 

! public routines
PUBLIC force9, reord, stack

CONTAINS

! **********************************************************            
!   FORCE9D  ORBIT9D                                                    
!                                                                       
!       general purpose right hand side with J2 and relativistic corr.  
!                                                                       
!   (vers. N+M bodies, 3d, no regular., nvz var.eq., closapp control)   
! **********************************************************            
SUBROUTINE force9(x,v,t,f,nd,idc,xxpla,ips,imem) 
  USE planet_masses, ONLY: dmin
  USE yark_pert, ONLY:  sec_nong9
  USE dyn_param, ONLY: iyark
  INCLUDE 'comnbo.h90' 
!  INCLUDE 'comdis.h90' 
! **********************************************************            
! nvar must be 6*(nbod-1+na+nvz); nvar2=nvar/2                          
! the velocity is available to allow for                                
! a velocity dependent right hand side; not used here                   
! warning: x begins with first planet; the coordinates of the Sun       
! are not among the dynamical variables. To better undertand the        
! allocation of the coordinates in the stack x, see stack, reord below  
! INPUT
  INTEGER, INTENT(IN) :: nd 
  DOUBLE PRECISION, INTENT(IN) :: x(nd),v(nd),t
  INTEGER, INTENT(IN) :: ips, imem ! for storage, so far dummy
! OUTPUT
  DOUBLE PRECISION, INTENT(OUT) :: f(nd)
! close approach data                                                   
  INTEGER, INTENT(OUT) :: idc 
  DOUBLE PRECISION, INTENT(OUT) :: xxpla(6) 
! END INTERFACE
  DOUBLE PRECISION dy(3),dyy(3,norbx),dd3(norbx),d2(norbx)
  DOUBLE PRECISION, DIMENSION(3):: s, sv, xb, vb, secacc
! variables added for J2 and relativistic computation                   
  DOUBLE PRECISION dr3(norbx),z2,x2y2,dr4,cj2r2,x2,y2,cj2r9,sc2r6,r6 
  DOUBLE PRECISION d2udx(3,3) 
! indexes for addresses                                                 
  INTEGER nvar,nvar2,norb,nvz1,nvz2,ia,iv,jad,ja,j1,j,jj 
! scalar temporaries                                                    
  DOUBLE PRECISION sum,st,d5,st1,d3 
! address precomputations                                               
  nvar=6*(nbod-1+na+nvz) 
  nvar2=nd 
  IF(nvar2*2.ne.nvar)THEN 
     WRITE(*,*)' force9d: nvar,nvar2,nbod,na,nvz=',nvar,nvar2,nbod,na,nvz
     STOP 
  ENDIF
  norb=nbod-1+na 
! close approach monitor                                                
  idc=0 
! first and last variational equation                                   
  nvz1=nbod 
  nvz2=nbod-1+nvz 
  ia=3*nbod-3 
  iv=ia+3*na 
! position of the sun in the barycentric system                         
!  $$s=-\sum_{j=2}^{nbod} \frac{m_j}{m_1} x_j $$                        
! (nbod-1)* (3 M + 3 S) unrolled inner loop                             
  s=0.d0 
  DO j=1,nbod-1 
     s(1:3)=s(1:3)-rm(j)*x(3*j-2:3*j) 
  ENDDO 
  IF(iyark.eq.3)THEN
     sv=0.d0 
     DO j=1,nbod-1 
        sv(1:3)=sv(1:3)-rm(j)*v(3*j-2:3*j) 
     ENDDO
  ENDIF 
! acceleration from the sun                                             
!  $$f^\circ_j=\frac{G m_1}{|x_j-x_1|^3}(x_1-x_j) $$                    
! (nbod-1+na)(8 M + 5 S + 1 SQRT)  unrolled inner loop                  
!$DIR NO_RECURRENCE                                                     
  DO 33 j=1,norb 
     st=s(1)-x(3*j-2) 
     sum=st*st 
     dyy(1,j)=st 
     st=s(2)-x(3*j-1) 
     sum=st*st+sum 
     dyy(2,j)=st 
     st=s(3)-x(3*j) 
     sum=st*st+sum 
     dyy(3,j)=st 
     d2(j)=sum 
     dr3(j)=1.d0/(sum*sqrt(sum)) 
     dd3(j)=gm(1)*dr3(j) 
     f(3*j-2)=dd3(j)*dyy(1,j) 
     f(3*j-1)=dd3(j)*dyy(2,j) 
     f(3*j)=dd3(j)*dyy(3,j) 
33 ENDDO
! Yarkovsky effect
  IF(iyark.eq.3.or.iyark.eq.4)THEN
     DO jj=1,na
        j=nbod+jj-1 ! loop on asteroid only
        xb=x(3*j-2:3*j)
        vb=v(3*j-2:3*j)
        CALL sec_nong9(xb,vb,s,sv,secacc,iyark,dadt9(jj))
        f(3*j-2:3*j)=f(3*j-2:3*j)+secacc
     ENDDO
  ENDIF
  IF(irelj2.gt.0)THEN
! J2 and relativistic corrections                                       
! from Nobili et al., A&A 1989 vol. 210 pag313-336; page 316            
     DO 34 j=1,norb 
        dr4=1.d0/d2(j)**2 
        x2y2=dyy(1,j)**2+dyy(2,j)**2 
        z2=dyy(3,j)**2 
        cj2r2=cj2*dr3(j)*3.d0 
        f(3*j-2)=f(3*j-2)+dyy(1,j)*(                                    &
     &    cj2r2*(2.d0*z2-0.5d0*x2y2)-                                   &
     &    2.d0*scw)*dr4                                                 
        f(3*j-1)=f(3*j-1)+dyy(2,j)*(                                    &
     &    cj2r2*(2.d0*z2-0.5d0*x2y2)-                                   &
     &    2.d0*scw)*dr4                                                 
        f(3*j)=f(3*j)+dyy(3,j)*(                                        &
     &    cj2r2*(z2-1.5d0*x2y2)-                                        &
     &    2.d0*scw)*dr4                                                 
34   ENDDO
  ENDIF
! variational equations for all asteroids 
   IF(irelj2.eq.0)THEN
! two body formula
! $$(\derparz{f^\circ_j}{x_j})_{kh}=-\frac{Gm_1}{|x_j-x_1|^3}\delta_{kh}
!   +\frac{3Gm_1}{|x_j-x_1|^5}(x_1-x_j)_k(x_1-x_j)_h$$                  
! nvz*(14 M + 5 S)  
      DO 3 j=nvz1,nvz2 
         jad=iv+3*(j-nvz1) 
         d5=3.d0*dd3(j)/d2(j) 
         st=dyy(1,j)*x(jad+1)+dyy(2,j)*x(jad+2)+dyy(3,j)*x(jad+3) 
         f(1+jad)=d5*st*dyy(1,j)-dd3(j)*x(1+jad) 
         f(2+jad)=d5*st*dyy(2,j)-dd3(j)*x(2+jad) 
         f(3+jad)=d5*st*dyy(3,j)-dd3(j)*x(3+jad) 
3     ENDDO
   ELSE            
! relativistic and J2 correction included
     DO 30 j=nvz1,nvz2 
!ccc       if(j.ge.nvz1.and.j.le.nvz2)then                              
! used in var. eq. for unperturbed orbit                                
        jad=iv+3*(j-nvz1) 
        d5=3.d0*dd3(j)/d2(j) 
        st=dyy(1,j)*x(jad+1)+dyy(2,j)*x(jad+2)+dyy(3,j)*x(jad+3) 
! used in the variational equations for J2, rel                         
        r6=dr3(j)*dr3(j) 
        x2=dyy(1,j)*dyy(1,j) 
        y2=dyy(2,j)*dyy(2,j) 
        z2=dyy(3,j)*dyy(3,j) 
        cj2r9=-3.d0*cj2*dr3(j)*r6 
        sc2r6=2.d0*scw*r6 
        d2udx(1,1)=x2*(cj2r9*(2.d0*x2+1.5d0*y2-13.5d0*z2)-3.d0*sc2r6)+  &
     &             y2*(cj2r9*(1.5d0*z2-0.5d0*y2)+sc2r6)+                &
     &             z2*(cj2r9*2.d0*z2+sc2r6)                             
        d2udx(2,2)=x2*(cj2r9*(1.5d0*z2-0.5d0*x2+1.5d0*y2)+sc2r6)+       &
     &             y2*(cj2r9*(2.d0*y2-13.5d0*z2)-3.d0*sc2r6)+           &
     &             z2*(cj2r9*2.d0*z2+sc2r6)                             
        d2udx(3,3)=x2*(cj2r9*(12.d0*z2-1.5d0*x2-3.d0*y2)+sc2r6)+        &
     &             y2*(cj2r9*(12.d0*z2-1.5d0*y2)+sc2r6)+                &
     &             z2*(-cj2r9*4.d0*z2-3.d0*sc2r6)                       
        d2udx(1,2)=dyy(1,j)*dyy(2,j)*(cj2r9*(2.5d0*x2+2.5*y2-15.d0*z2)  &
     &             -4.d0*sc2r6)                                         
        d2udx(2,1)=d2udx(1,2) 
        d2udx(1,3)=dyy(1,j)*dyy(3,j)*(cj2r9*(7.5d0*x2+7.5*y2-10.d0*z2)  &
     &             -4.d0*sc2r6)                                         
        d2udx(3,1)=d2udx(1,3) 
        d2udx(2,3)=dyy(2,j)*dyy(3,j)*(cj2r9*(7.5d0*x2+7.5*y2-10.d0*z2)  &
     &             -4.d0*sc2r6)                                         
        d2udx(3,2)=d2udx(2,3) 
! right hand side of variational equations                              
        f(1+jad)=d5*st*dyy(1,j)-dd3(j)*x(1+jad)+                        &
     &    d2udx(1,1)*x(1+jad)+d2udx(1,2)*x(2+jad)+                      &
     &    d2udx(1,3)*x(3+jad)                                           
        f(2+jad)=d5*st*dyy(2,j)-dd3(j)*x(2+jad)+                        &
     &    d2udx(2,1)*x(1+jad)+d2udx(2,2)*x(2+jad)+                      &
     &    d2udx(2,3)*x(3+jad)                                           
        f(3+jad)=d5*st*dyy(3,j)-dd3(j)*x(3+jad)+                        &
     &    d2udx(3,1)*x(1+jad)+d2udx(3,2)*x(2+jad)+                      &
     &    d2udx(3,3)*x(3+jad)                                           
!ccc       endif                                                        
30   ENDDO
  ENDIF
! iteration on the number of planets                                    
  DO 6 j=1,nbod-1 
! planet-planet interaction                                             
!  $$ f_{ji}=\frac{x_j-x_i}{|x_j-x_i|^3}\ \ \ for\ i<j$$                
!  $$f_j=f^\circ_j+\sum_{i<j}Gm_i f_{ji} + \sum_{i>j}Gm_i f_{ij}$$      
! (nbod-1)*(nbod-2)/2 * (13 M +11 S + 1 SQRT) unrolled inner loop       
!ccccC$DIR NO_RECURRENCE                                                
     DO 7 j1=1,j-1 
! vector differences and distances                                      
        st=x(3*j-2)-x(3*j1-2) 
        sum=st*st 
        dy(1)=st 
        st=x(3*j-1)-x(3*j1-1) 
        sum=sum+st*st 
        dy(2)=st 
        st=x(3*j)-x(3*j1) 
        sum=sum+st*st 
        dy(3)=st 
        d3=1.d0/(sum*sqrt(sum)) 
! mutual perturbations                                                  
        st=d3*gm(j+1) 
        st1=d3*gm(j1+1) 
        f(3*j1-2)=f(3*j1-2)+st*dy(1) 
        f(3*j-2)=f(3*j-2)-st1*dy(1) 
        f(3*j1-1)=f(3*j1-1)+st*dy(2) 
        f(3*j-1)=f(3*j-1)-st1*dy(2) 
        f(3*j1)=f(3*j1)+st*dy(3) 
        f(3*j)=f(3*j)-st1*dy(3) 
7    ENDDO
! planet-asteroid interaction                                           
! na*(nbod-1)*(8 M + 8 S + 1 SQRT) unrolled inner loop                  
!$DIR NO_RECURRENCE                                                     
     DO 8 ja=1,na 
! vector differences and distances                                      
        st=x(3*j-2)-x(3*ja-2+ia) 
        sum=st*st 
        dy(1)=st 
        st=x(3*j-1)-x(3*ja-1+ia) 
        sum=sum+st*st 
        dy(2)=st 
        st=x(3*j)-x(3*ja+ia) 
        sum=sum+st*st 
        dy(3)=st 
        d3=gm(j+1)/(sum*sqrt(sum)) 
! perturbations                                                         
        f(3*ja-2+ia)=f(3*ja-2+ia)+dy(1)*d3 
        f(3*ja-1+ia)=f(3*ja-1+ia)+dy(2)*d3 
        f(3*ja+ia)=f(3*ja+ia)+dy(3)*d3 
!  distance control                                                     
        if(sum.le.dmin(j))then 
           if(idc.eq.0)then 
              idc=j+(nbod-1)*ja 
              xxpla(1:3)=x(3*j-2:3*j)
              xxpla(4:6)=v(3*j-2:3*j)
           else 
              write(*,*)'force: double encounter ',t,idc,j,ia 
           endif
        endif
! variational equations for some asteroids                              
! nvz*(nbod-1)*(14 M + 8 S)                                             
!c         if(ja.le.nvz)then                                            
        jad=iv+3*(ja-1) 
        d5=3.d0*d3/sum 
        st=dy(1)*x(jad+1)+dy(2)*x(jad+2)+dy(3)*x(jad+3) 
        f(1+jad)=d5*st*dy(1)-d3*x(1+jad)+f(1+jad) 
        f(2+jad)=d5*st*dy(2)-d3*x(2+jad)+f(2+jad) 
        f(3+jad)=d5*st*dy(3)-d3*x(3+jad)+f(3+jad) 
!c         endif                                                        
    8 ENDDO 
! end loop on number of planets                                         
6 ENDDO
! =================================================                     
!  flop summary                                                         
!                                                                       
! do 2: (nbod-1)* 8                                                     
!                                                                       
! do 3 :(nbod-1+na)(13 + SQRT) + nvz*19                                 
!                                                                       
! do 7 :(nbod-1)*(nbod-2)/2 * (24 + SQRT)                               
!                                                                       
! do 8: na*(nbod-1)*(16 + SQRT) + nvz*(nbod-1)*22                       
!                                                                       
! tot: nvz*(nbod*22 -3) + na*(nbod*(13+SQRT)-3) +                       
! (nbod-1)(nbod-2)/2 * (24 + SQRT) + (nbod-1)*8                         
END SUBROUTINE force9
! *********************************************************             
!  {\bf reord} ORB8V output: reordering of the arrays                   
SUBROUTINE reord(y2,nvar,yp,npla,ya,na,yv,nvz,ndim,ndim2) 
  IMPLICIT NONE 
  INTEGER, INTENT(IN) :: nvar,ndim,ndim2,na,nvz,npla
  DOUBLE PRECISION, INTENT(IN) :: y2(nvar)
  DOUBLE PRECISION, INTENT(OUT) :: yp(ndim2,npla),ya(ndim2,na),yv(ndim2,nvz) 
! END INTERFACE  
  INTEGER nbod,nvar2                
  integer jj,j,i 
  nvar2=nvar/2 
  nbod=npla+1 
!  planets: reordering of the arrays                                    
  DO 11 j=1,npla 
     jj=3*j-3 
     DO i=1,ndim 
        yp(i+ndim,j)=y2(nvar2+jj+i) 
        yp(i,j)=y2(i+jj)
     ENDDO
11 ENDDO
!  asteroids: reordering of the arrays                                  
  DO 12 j=1,na 
     jj=3*(nbod-1+j)-3 
     DO i=1,ndim 
        ya(i+ndim,j)=y2(nvar2+jj+i) 
        ya(i,j)=y2(i+jj)
     ENDDO
12 ENDDO
!  variational equations                                                
  DO 20 j=1,nvz 
     jj=3*(nbod-1+na+j)-3 
     DO i=1,ndim 
        yv(i+ndim,j)=y2(nvar2+jj+i) 
        yv(i,j)=y2(i+jj)
     ENDDO
20 ENDDO
END SUBROUTINE reord
! *********************************************************             
!  {\bf stack} ORB8V initialisation: reordering of the arrays           
SUBROUTINE stack(y1,nvar,yp,npla,ya,na,yv,nvz,ndim,ndim2) 
  IMPLICIT NONE 
  INTEGER, INTENT(IN) ::  nvar,npla,na,nvz,ndim,ndim2
  DOUBLE PRECISION, INTENT(IN) :: yp(ndim2,npla),ya(ndim2,na),yv(ndim2,nvz)
  DOUBLE PRECISION, INTENT(OUT) :: y1(nvar)
  ! END INTERFACE                                               
  integer jj,j,i,nvar2 
  nvar2=nvar/2 
!  planets                                                              
  DO 1 j=1,npla 
     jj=3*j-3 
     DO i=1,ndim 
        y1(nvar2+jj+i)=yp(i+ndim,j) 
        y1(i+jj)=yp(i,j)
     ENDDO
1 ENDDO
! asteroids                                                             
  DO 2 j=1,na 
     jj=3*(npla+j)-3 
     DO i=1,ndim 
        y1(nvar2+jj+i)=ya(i+ndim,j) 
        y1(i+jj)=ya(i,j)
     ENDDO
2 ENDDO
! variational equations                                                 
  DO 3 j=1,nvz 
     jj=3*(npla+na+j)-3 
     DO i=1,ndim 
        y1(nvar2+jj+i)=yv(i+ndim,j) 
        y1(i+jj)=yv(i,j)
     ENDDO
3 ENDDO
END SUBROUTINE stack


END MODULE force9d
