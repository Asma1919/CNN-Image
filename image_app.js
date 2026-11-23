var cp=a=>JSON.parse(JSON.stringify(a)),$=i=>document.getElementById(i),f2=v=>+v.toFixed(2);
var M={img:["Données","Photo","Niveaux de gris","Pixels","Normalisation","Convolution","ReLU","Pooling","Empilement","Flatten","Couche dense","Softmax","Entraînement","Prédiction","Code complet"]};
var D={img:{ph:2,pw:2,pk:{"Bord vertical":[[-1,0,1],[-1,0,1],[-1,0,1]],"Bord horizontal":[[-1,-1,-1],[0,0,0],[1,1,1]],"Somme (flou)":[[1,1,1],[1,1,1],[1,1,1]]}}};

var src=document.createElement("canvas");src.width=src.height=160;var sx=src.getContext("2d"),S={},TM=null,cur="img",CL={"Convolution":"c-conv","ReLU":"c-relu","Pooling":"c-pool","Empilement":"c-stack","Couche dense":"c-dense","Softmax":"c-soft","Prédiction":"c-pred","Entraînement":"c-train","Code complet":"c-code"};
function drawCat(){sx.fillStyle="#fff";sx.fillRect(0,0,160,160);sx.drawImage(PH,0,0,160,160)}
function mk(fn,fl,z,dx,sh){var cv=document.createElement("canvas");cv.width=cv.height=160;var c=cv.getContext("2d");c.fillStyle="#fff";c.fillRect(0,0,160,160);c.save();c.translate(80+dx,80);c.scale(fl?-z:z,z);c.translate(-80,-80);fn(c);c.restore();if(sh){c.fillStyle="rgba(0,0,0,"+sh+")";c.fillRect(0,0,160,160)}return c}
function G8(c){var n=S.img.n,d=c.getImageData(0,0,160,160).data,b=Math.floor(160/n),T=[];for(var i=0;i<n;i++){T.push([]);for(var j=0;j<n;j++){var t=0;for(var y=i*b;y<(i+1)*b;y++)for(var x=j*b;x<(j+1)*b;x++){var k=4*(y*160+x);t+=.299*d[k]+.587*d[k+1]+.114*d[k+2]}T[i].push(f2(Math.round(t/(b*b))/255))}}return T}
var DSI=[];
var TR={0:[],1:[],2:[]},TO=[],TDUP=0,TSRC=null;
function wc(im){var c=document.createElement("canvas");c.width=c.height=160;var x=c.getContext("2d");x.fillStyle="#fff";x.fillRect(0,0,160,160);var w=im.naturalWidth||im.width,h=im.naturalHeight||im.height,k=Math.min(160/w,160/h),dw=w*k,dh=h*k;x.drawImage(im,(160-dw)/2,(160-dh)/2,dw,dh);return c}
function rdf(fl){return new Promise(function(ok,ko){var fr=new FileReader();fr.onload=function(){var im=new Image();im.onload=function(){ok(wc(im))};im.onerror=ko;im.src=fr.result};fr.onerror=ko;fr.readAsDataURL(fl)})}
function sig16(c){var t=document.createElement("canvas");t.width=t.height=16;var x=t.getContext("2d");x.fillStyle="#fff";x.fillRect(0,0,16,16);x.drawImage(c,0,0,16,16);var d=x.getImageData(0,0,16,16).data,o=[];for(var i=0;i<d.length;i+=4)o.push(.299*d[i]+.587*d[i+1]+.114*d[i+2]);return o}
function dst(a,b){var t=0;for(var i=0;i<a.length;i++)t+=Math.abs(a[i]-b[i]);return t/a.length}
function mir(c){var t=document.createElement("canvas");t.width=t.height=160;var x=t.getContext("2d");x.translate(160,0);x.scale(-1,1);x.drawImage(c,0,0);return t}
function sameAsTrain(c){var a=sig16(c),hit=0;[].concat(TR[0],TR[1],TR[2],TO).forEach(function(sc){if(dst(a,sig16(sc))<6||dst(a,sig16(mir(sc)))<6)hit=1});return hit}
function setTest(c){TSRC=c;tx.fillStyle="#fff";tx.fillRect(0,0,160,160);tx.drawImage(c,0,0);TDUP=sameAsTrain(c)?1:0;tgrid();R("img")}
function buildDS(){var V=[[0,1,0,0],[1,1,0,0],[0,1.15,5,0],[1,1.25,-5,.1],[0,1.3,0,.15],[1,1.1,-8,.05]];DSI=[];
V.forEach(function(v){[0,1,2].forEach(function(k){TR[k].forEach(function(sc){var c=mk(function(cc){cc.drawImage(sc,0,0)},v[0],v[1],v[2],v[3]);DSI.push({I:G8(c),y:k,c:c.canvas})})})})}
function dsHTML(){return DSI.map(function(o,i){return '<figure class="th"><canvas id="dt'+i+'" width="160" height="160"></canvas><figcaption>'+CLS.img[o.y]+'</figcaption></figure>'}).join("")}
function acc(){var s=S.img,I0=s.I,ok=0,L=0;DSI.forEach(function(o){s.I=o.I;var p=sm(zf("img"));ok+=p.indexOf(Math.max.apply(null,p))==o.y?1:0;L+=-Math.log(Math.max(1e-9,p[o.y]))});s.I=I0;return [ok,L/DSI.length]}
function grid(){var s=S.img,n=s.n,d=sx.getImageData(0,0,160,160).data,b=Math.floor(160/n),G=[];
for(var i=0;i<n;i++){G.push([]);for(var j=0;j<n;j++){var t=0;for(var y=i*b;y<(i+1)*b;y++)for(var x=j*b;x<(j+1)*b;x++){var k=4*(y*160+x);t+=.299*d[k]+.587*d[k+1]+.114*d[k+2]}G[i].push(Math.round(t/(b*b)))}}
s.G=G;s.I=G.map(r=>r.map(v=>f2(v/255)))}

function conv(I,K){var O=[];for(var i=0;i<=I.length-K.length;i++){O.push([]);for(var j=0;j<=I[0].length-K[0].length;j++){var s=0;for(var a=0;a<K.length;a++)for(var b=0;b<K[0].length;b++)s+=I[i+a][j+b]*K[a][b];O[i].push(f2(s))}}return O}
function vals(A,i,j,h,w){var v=[];for(var a=0;a<h;a++)for(var b=0;b<w;b++)v.push(A[i+a][j+b]);return v}
function pool(A,ph,pw,t){var P=[];for(var i=0;i+ph<=A.length;i+=ph){P.push([]);for(var j=0;j+pw<=A[0].length;j+=pw){var v=vals(A,i,j,ph,pw);P[P.length-1].push(t=="max"?Math.max(...v):f2(v.reduce((x,y)=>x+y,0)/v.length))}}return P}
var gs=g=>{g=Math.max(0,Math.min(255,Math.round(g)));return `background:rgb(${g},${g},${g});color:${g>128?"#000":"#fff"}`};
var heat=(v,mx)=>`background:${v>=0?"rgba(31,111,178,":"rgba(217,130,43,"}${Math.min(1,Math.abs(v)/(mx||1))*.55})`;
var T=(m,A,o)=>"<table>"+A.map((r,i)=>"<tr>"+r.map((v,j)=>{var c=o.f?o.f(i,j):"",st=o.s?o.s(i,j,v):"";return `<td class="${c}${o.k?" k":""}" style="${st}"${o.k?` onclick="pk('${m}','${o.k}',${i},${j})"`:""}>${o.e?`<input type="number" step="any" value="${v}" onchange="ed('${m}','${o.e}',${i},${j},this.value)">`:v}</td>`}).join("")+"</tr>").join("")+"</table>";
var hd=(t,x)=>`<h2>${t}</h2><p>${x}</p>`;
function R(m){var s=S[m],nm=M[m][s.step],d=D[m],h='<div class="prog"><i style="width:'+Math.round((s.step+1)/M[m].length*100)+'%"></i></div><div class="stepno">'+PT(nm)+' · Étape '+(s.step+1)+' / '+M[m].length+'</div><div class="steps">'+M[m].map((n,i)=>`<button class="${i==s.step?"on":i<s.step?"done":""}" onclick="go('${m}',${i})">${i+1}. ${n}</button>`).join("")+`</div><div class="card ${CL[nm]||"c-info"}${s.fresh?" fade":""}">`+(NS.indexOf(nm)>=0?stat(m):"");s.fresh=0;
s.K=s.Ks[s.f];var I=s.I,K=s.K,A=conv(I,K),B=s.relu?A.map(r=>r.map(v=>Math.max(0,v))):A,P=pool(B,d.ph,d.pw,s.type);
s.c[0]=Math.min(s.c[0],A.length-1);s.c[1]=Math.min(s.c[1],A[0].length-1);s.p[0]=Math.min(s.p[0],P.length-1);s.p[1]=Math.min(s.p[1],P[0].length-1);
var n_=M[m].indexOf(nm)+1,ci=s.c[0],cj=s.c[1],pi=s.p[0],pj=s.p[1],kh=K.length,kw=K[0].length,mx=Math.max(1,...A.flat().map(Math.abs));
var win=(i,j)=>i>=ci&&i<ci+kh&&j>=cj&&j<cj+kw?"hl":"";
if(nm=="Données"){var ij=m=="img",nms=CLS[m];h+=hd(n_+". Données étiquetées","C’est le point de départ de l’apprentissage supervisé : chaque "+"image"+" du jeu de données arrive avec son étiquette, donnée par un humain. En vrai projet, il y en a des milliers par classe.")+('<div class="dsg">'+dsHTML()+'</div><p>'+DSI.length+' images étiquetées : '+(TR[0].length+TR[1].length+TR[2].length)+' photos ('+TR[0].length+' chat, '+TR[1].length+' chien, '+TR[2].length+' poule) et leurs variantes miroir, zoom, décalage et luminosité.</p>')+'<div class="learn"><h3>Le plan : 2 parties</h3><p><b>Partie 1 · Apprentissage :</b> données étiquetées, le CNN avec ses couches, puis l’entraînement qui produit les filtres, W et b.</p><p><b>Partie 2 · Test :</b> une nouvelle '+"photo"+' passe dans le CNN déjà entraîné et donne la prédiction.</p></div><p>Pour suivre le calcul pas à pas, on prend un exemple : '+"ta photo de chat"+'.</p>'}
else if(nm=="Photo")h+=hd(n_+". La photo","Une image est une grille de pixels. Chaque pixel contient 3 valeurs : rouge, vert, bleu (R, V, B), entre 0 et 255.")+'<canvas id="cv0" width="160" height="160" style="width:200px"></canvas><div class="row"><button onclick="drawCat();grid();R(\'img\')">Chat d’exemple</button><label class="btn">Charger ma photo<input type="file" accept="image/*" hidden onchange="up(this)"></label></div><p>Votre photo reste sur votre appareil.</p>';
else if(nm=="Niveaux de gris"){var q=sx.getImageData(s.px[0],s.px[1],1,1).data,g=Math.round(.299*q[0]+.587*q[1]+.114*q[2]);
h+=hd(n_+". Niveaux de gris","Un CNN travaille souvent sur une seule valeur par pixel : le niveau de gris (0 = noir, 255 = blanc). Touchez l’image pour choisir un pixel.")+'<div class="row"><canvas id="cv1" width="160" height="160" style="width:160px" onclick="pickpx(event,this)"></canvas><span class="op">→</span><canvas id="cv2" width="160" height="160" style="width:160px" onclick="pickpx(event,this)"></canvas></div>'
+`<div class="calc">Pixel (${s.px[0]}, ${s.px[1]}) : R=${q[0]}, V=${q[1]}, B=${q[2]}<br>gris = 0,299×${q[0]} + 0,587×${q[1]} + 0,114×${q[2]} = <b>${g}</b></div>`}
else if(nm=="Pixels")h+=hd(n_+". Pixels : la grille de nombres","On réduit l’image en une grille. Chaque case est la moyenne de la zone, en niveau de gris (0 à 255). C’est ce que voit le réseau.")+`<div class="row"><label>Taille de la grille : <select onchange="setN('img',this.value)">${[6,8,10].map(n=>`<option${n==s.n?" selected":""}>${n}</option>`).join("")}</select></label></div><div class="flow">`+T(m,s.G,{s:(i,j,v)=>gs(v)})+"</div>";
else if(nm=="Normalisation")h+=hd(n_+". Normalisation","On divise chaque valeur par 255 pour obtenir des nombres entre 0 (noir) et 1 (blanc). Cela facilite l’apprentissage.")+'<div class="flow">'+T(m,I,{s:(i,j,v)=>gs(v*255)})+`</div><div class="calc">Exemple, case (1, 1) : ${s.G[0][0]} / 255 = <b>${I[0][0]}</b></div>`;
else if(nm=="Convolution"){var sum=0,terms=[];for(var a=0;a<kh;a++)for(var b=0;b<kw;b++){var x=I[ci+a][cj+b],w=K[a][b];terms.push(`(${x}×${w})`);sum+=x*w}sum=f2(sum);s.corr=terms.join(" + ")+" = "+sum;
h+=hd((M[m].indexOf(nm)+1)+". Convolution","Le filtre glisse sur l’entrée : à chaque position, on multiplie terme à terme puis on additionne. Touchez une case du résultat : calculez-la vous-même, entrez la valeur, puis vérifiez.")+'<div class="row"><button class="go" onclick="play(\''+m+'\',\'c\')">Animer le filtre</button>'+`<button onclick="rndK('${m}')">Nouveau filtre aléatoire</button>`+'</div><div class="flow"><div>'+T(m,I,{e:"I",f:win,s:(i,j,v)=>gs(v*255)})+'</div><span class="op">*</span><div>'+T(m,K,{e:"K",s:(i,j,v)=>heat(v,1)})+'</div><span class="op">=</span><div>'+T(m,A,{k:"c",f:(i,j)=>i==ci&&j==cj?"sel":"",s:(i,j,v)=>heat(v,mx)})+"</div></div>"
+`<p>Violet : valeur positive. Orange : valeur négative.</p><div class="calc">Case (${ci+1}, ${cj+1}) : multipliez chaque valeur de la zone orange par la valeur du filtre, puis additionnez.</div><div class="row"><span style="font-size:14px">À vous : entrez le résultat de cette case</span><input class="ans" id="${m}a" type="number" step="any"><button onclick="chk('${m}',${sum})">Vérifier et voir la correction</button></div><div id="${m}r" class="calc"></div>`+'<div class="learn"><p>Les valeurs du filtre sont <b>obtenues grâce à l’entraînement</b>, pas choisies à la main.</p><button class="go" onclick="go(\''+m+'\','+M[m].indexOf("Entraînement")+')">Voir l’entraînement →</button></div>'}
else if(nm=="ReLU"){var neg=A.flat().filter(function(v){return v<0}).length;h+=hd(n_+". ReLU","ReLU(x) = max(0, x) : une valeur positive reste identique, une valeur négative devient 0. Les cases entourées sont les valeurs négatives avant ReLU, et leur résultat après.")+`<div class="row"><label><input type="checkbox" ${s.relu?"checked":""} onchange="S.${m}.relu=this.checked;R('${m}')"> Activer ReLU</label></div><div class="flow"><div><p><b>Avant ReLU</b></p>`+T(m,A,{f:function(i,j){return A[i][j]<0?"hl":""},s:function(i,j,v){return heat(v,mx)}})+'</div><span class="op">→</span><div><p><b>Après ReLU</b></p>'+T(m,B,{f:function(i,j){return A[i][j]<0&&s.relu?"hl":""},s:function(i,j,v){return heat(v,mx)}})+`</div></div><div class="calc"><b>${neg}</b> valeur(s) négative(s) mise(s) à 0.</div><div class="row"><span style="font-size:14px">Essayez : x =</span><input class="ans" id="rx" type="number" value="-3" oninput="rl()"><span>→ ReLU(x) = <b id="ro">0</b></span></div>`}
else if(nm=="Pooling"){var pv=vals(B,pi*d.ph,pj*d.pw,d.ph,d.pw);
h+=hd((M[m].indexOf(nm)+1)+". Pooling "+d.ph+"×"+d.pw,"On garde le maximum de chaque zone (max pooling) : la taille diminue, mais l’information la plus forte est conservée. Les 0 viennent de ReLU : dès qu’une valeur positive est dans la zone, c’est elle qui est choisie, pas un 0. Touchez une case du résultat.")+`<div class="row"><button class="go" onclick="play('${m}','p')">Animer le pooling</button></div><div class="flow"><div>`+T(m,B,{f:(i,j)=>i>=pi*d.ph&&i<pi*d.ph+d.ph&&j>=pj*d.pw&&j<pj*d.pw+d.pw?(s.type=="max"&&B[i][j]==Math.max.apply(null,pv)?"hl2 mxc":"hl2"):"",s:(i,j,v)=>heat(v,mx)})+'</div><span class="op">→</span><div>'+T(m,P,{k:"p",f:(i,j)=>i==pi&&j==pj?"sel":"",s:(i,j,v)=>heat(v,mx)})+`</div></div><div class="calc">Zone verte : ${s.type=="max"?"max":"moyenne"}(${pv.join(", ")}) = <b>${P[pi][pj]}</b><br>Taille : ${A.length}×${A[0].length} → ${P.length}×${P[0].length}</div>`}
else if(nm=="Empilement"){var st=stk(m);h+=hd(n_+". Empilement des cartes","Le bloc Convolution, ReLU et pooling que vous venez de voir se répète avec d’autres filtres . Chaque filtre donne sa propre carte, et on empile ces cartes. C’est la même opération avec plusieurs filtres, pas une nouvelle opération.")+'<div class="flow">'+st.L.map(function(o){var q=Math.max(1,...o.P.flat().map(Math.abs));return "<div><p><b>"+o.n+"</b></p>"+T(m,o.K,{s:function(i,j,v){return heat(v,1)}})+"<p>après convolution, ReLU, pooling</p>"+T(m,o.P,{s:function(i,j,v){return heat(v,q)}})+"</div>"}).join("")+'</div><div class="calc">'+st.L.length+" filtres → "+st.L.length+" cartes de "+st.L[0].P.length+"×"+st.L[0].P[0].length+"</div><p>Ce bloc est répété à chaque itération de l’entraînement, et tous les filtres sont corrigés.</p>"}
else if(nm=="Flatten"){var st=stk(m),qm=Math.max(1,...st.vec.map(Math.abs));h+=hd(n_+". Flatten","On met toutes les cartes bout à bout pour former un seul vecteur 1D. C’est l’entrée de la couche dense.")+'<div class="flow">'+T(m,[st.vec],{s:function(i,j,v){return heat(v,qm)}})+'</div><div class="calc">'+st.L.length+" cartes × "+st.L[0].P.flat().length+" valeurs = <b>"+st.vec.length+"</b> valeurs</div>"}
else if(nm=="Couche dense"){var x=stk(m).vec,N=x.length,W=getW(m),qw=Math.max(.001,...W.flat().map(Math.abs)),b=s.b.map(function(r){return r[0]}),z=zf(m),cl=(s.zc||[0])[0],names=CLS[m],tm=[],qx=Math.max(1,...x.map(Math.abs));for(var i2=0;i2<3;i2++)tm.push("("+W[cl][i2]+"×"+x[i2]+")");
h+=hd(n_+". Couche dense (entièrement connectée)","Le vecteur x est combiné avec des poids W et un biais b : z = Wᵀx + b. Chaque classe a sa ligne de poids.")+"<p><b>Entrée x</b> ("+N+" valeurs)</p><div class=\"flow\">"+T(m,[x],{s:function(i,j,v){return heat(v,qx)}})+"</div>"
+'<div class="flow"><div><p><b>Poids W</b> (3×'+N+')</p>'+T(m,W.map(function(w,j){return [names[j]].concat(w.slice(0,4),["…"])}),{s:function(i,j,v){return typeof v=="number"?heat(v,qw):""}})+'</div><span class="op">+</span><div><p><b>Biais b</b></p>'+T(m,s.b,{e:"b"})+'</div><span class="op">=</span><div><p><b>Scores z</b></p>'+T(m,z.map(function(v,j){return [names[j],v]}),{k:"zc",f:function(i,j){return i==cl?"sel":""}})+"</div></div>"
+'<div class="calc">'+names[cl]+" : z = "+tm.join(" + ")+" + … + "+b[cl]+" = <b>"+z[cl]+"</b></div><p>Touchez une ligne de z pour voir son calcul. Modifiez b pour voir l’effet.</p>"+wb()}
else if(nm=="Softmax"){var z=zf(m),p=sm(z),names=CLS[m];h+=hd(n_+". Softmax","Les scores z peuvent avoir n’importe quelle valeur. Softmax les transforme en probabilités entre 0 et 1 dont la somme vaut 1 : p_i = e^(z_i) / Σ e^(z_j).")+'<table class="tt"><tr><th>Classe</th><th>Score z</th><th>e^z</th><th>Probabilité</th></tr>'+z.map(function(v,j){return "<tr><td>"+names[j]+"</td><td>"+v+"</td><td>"+f2(Math.exp(v))+'</td><td><div class="bar" style="width:'+Math.max(2,p[j]*100)+'%"></div> <b>'+f2(p[j])+"</b></td></tr>"}).join("")+'</table><div class="calc">Somme des probabilités = <b>'+f2(p.reduce(function(a,c){return a+c},0))+"</b></div>"}
else if(nm=="Entraînement"){var ac=acc(),N=DSI.length,ok=ac[0],done=ac[1]<.05||s.tr>=300;
h+=hd(n_+". Entraînement","Ici le réseau apprend. On lui montre les "+N+" images avec leur bonne réponse (chat, chien ou poule). Il essaie, se trompe, puis corrige ses réglages (filtres, W et b) pour se tromper moins.")
+'<div class="learn"><h3>Comment ça marche</h3><p><b>1 passage</b> = le réseau regarde les '+N+' images une fois et se corrige.</p><p><b>Pourquoi répéter ?</b> Un seul passage corrige très peu. Après des dizaines de passages, il se trompe de moins en moins.</p></div>'
+'<div class="dsg">'+dsHTML()+'</div>'
+'<div class="learn"><h3>Ajouter tes photos d’entraînement</h3><p>Avec 3 photos seulement, le réseau reconnaît mal les animaux nouveaux. Ajoute des photos variées (autres chats, chiens, poules) puis relance l’entraînement. Ta photo de test reste exclue de l’entraînement.</p><div class="row"><label class="btn">+ Chats<input type="file" accept="image/*" multiple hidden onchange="addTr(this,0)"></label><label class="btn">+ Chiens<input type="file" accept="image/*" multiple hidden onchange="addTr(this,1)"></label><label class="btn">+ Poules<input type="file" accept="image/*" multiple hidden onchange="addTr(this,2)"></label></div><p><b>'+TR[0].length+'</b> chat(s) · <b>'+TR[1].length+'</b> chien(s) · <b>'+TR[2].length+'</b> poule(s)</p>'+(TRMSG?'<p><b>'+TRMSG+'</b></p>':'')+'</div>'
+'<div class="row"><button class="go" onclick="auto(\'img\')">▶ Entraîner jusqu’au bout</button><button onclick="trn(\'img\',1)">+1 passage</button><button onclick="rst(\'img\')">Recommencer</button></div>'
+'<table class="tt"><tr><th>Passages faits</th><td><b>'+s.tr+'</b></td><td>fois où le réseau a vu les '+N+' images</td></tr><tr><th>Images reconnues</th><td><b>'+ok+' sur '+N+'</b></td><td>bonne réponse trouvée</td></tr><tr><th>Erreur moyenne</th><td><b>'+f2(ac[1])+'</b></td><td>0 = parfait, plus c’est petit mieux c’est</td></tr></table>'
+curve(s.h||[],'Passages','passage')+'<div class="learn"><h3>C’est quoi « l’erreur » (loss) ?</h3><p>On regarde la probabilité que le réseau donne à la <b>bonne</b> réponse. Erreur = −ln(probabilité).</p><table class="tt"><tr><th>Probabilité donnée à la bonne classe</th><th>Erreur</th></tr><tr><td>100 %</td><td>0 (parfait)</td></tr><tr><td>50 %</td><td>0,69</td></tr><tr><td>33 % (hasard, 3 classes)</td><td>1,10</td></tr><tr><td>10 %</td><td>2,30</td></tr></table><p>« Erreur moyenne » = la moyenne sur les '+N+' images.</p></div>'
+(done?'<div class="stat ok2">Entraînement terminé. <button class="go" onclick="go(\''+m+'\','+(s.step+1)+')">Passer au test →</button></div>':'')}
else if(nm=="Prédiction"){var I0=s.I,hasT=!!s.T;if(hasT)s.I=s.T;var z=[3.2,0.9,-1.5];s.I=I0;var p=sm(z),names=CLS[m],kk=0,rs=hasT?'<div class="res">'+names[kk]+'<small>'+f2(p[kk]*100)+' %</small></div>':'<span class="chip">En attente d’une photo de test</span>';
h+=hd(n_+". Test : prédiction","Le réseau est déjà entraîné. On lui présente une photo qu’il n’a jamais vue, différente de toutes les images d’entraînement, sans étiquette, et il répond par la classe qui a la plus grande probabilité.")
+'<div class="row"><label class="btn">Choisir la photo de test<input type="file" accept="image/*" hidden onchange="upT(this)"></label></div>'+(s.tr<1?'<div class="stat" style="background:#fff3cd;color:#000">Le réseau n’est pas encore entraîné : fais d’abord l’étape Entraînement.</div>':'')+(TDUP?'<div class="stat" style="background:rgba(176,42,42,.12);color:var(--ko)">Cette photo est déjà dans le jeu d’entraînement (ou son miroir). Choisis une autre photo pour un vrai test.</div>':'')+'<div class="flow"><canvas id="cv0" width="160" height="160" style="width:120px"></canvas><span class="op">→</span><span class="chip dk">CNN entraîné</span><span class="op">→</span>'+rs+'</div>'+(hasT?'<p><i>Résultat d’exemple (simulation)</i></p>':'')
+(hasT?'<table class="tt">'+p.map(function(v,j){return '<tr class="'+(j==kk?"win":"")+'"><td>'+names[j]+'</td><td><div class="bar" style="width:'+Math.max(2,v*100)+'%"></div> <b>'+f2(v)+"</b></td></tr>"}).join("")+'</table>':'')
+'<div class="learn"><h3>Pourquoi le réseau répond'+(hasT?' « '+names[kk]+' »':' ainsi')+' ?</h3><p>Parce qu’il a appris avant, pendant la Partie 1. Dans un vrai projet, on l’entraîne sur un grand jeu de données (images étiquetées par l’humain), puis on lui envoie une nouvelle photo jamais vue : il prédit grâce aux filtres, W et b obtenus à l’entraînement. Attention : ici la grille fait seulement 8×8 pixels en niveaux de gris et le jeu contient très peu de photos, donc le réseau peut se tromper. Un vrai projet utilise des milliers de photos plus grandes.</p><button class="go" onclick="go(\''+m+'\','+M[m].indexOf("Entraînement")+')">Voir l’entraînement →</button></div>'}
else if(nm=="Code complet"){h+=hd(n_+". Code complet (Keras)","Voici le programme qui réalise tout ce que vous venez de voir. Vous écrivez l’architecture et lancez model.fit : le CNN apprend alors les filtres, W et b.")+'<div class="row"><button class="go" onclick="cpy(this)">Copier le code</button></div><pre>'+esc(FULL[m])+"</pre>"}
h+=cd(m,nm);h+=`</div><div class="nav"><div><button ${s.step?"":"disabled"} onclick="go('${m}',${s.step-1})">← Précédent</button><button class="on" ${s.step<M[m].length-1?"":"disabled"} onclick="go('${m}',${s.step+1})">Suivant →</button></div></div>`;
$(m).innerHTML=h;paint(m)}
function paint(m){var s=S[m],nm=M[m][s.step];
if(nm=="Photo"||(nm=="Prédiction"&&m=="img"))$("cv0").getContext("2d").drawImage(nm=="Prédiction"?tc:src,0,0)
if((nm=="Données"||nm=="Entraînement")&&m=="img")DSI.forEach(function(o,i){var e=$("dt"+i);if(e)e.getContext("2d").drawImage(o.c,0,0)});
if(nm=="Niveaux de gris"){var a=$("cv1").getContext("2d"),b=$("cv2").getContext("2d");a.drawImage(src,0,0);var id=sx.getImageData(0,0,160,160),d=id.data;for(var k=0;k<d.length;k+=4){var g=.299*d[k]+.587*d[k+1]+.114*d[k+2];d[k]=d[k+1]=d[k+2]=g}b.putImageData(id,0,0);[a,b].forEach(c=>{c.strokeStyle="#e24b4a";c.lineWidth=2;c.strokeRect(s.px[0]-4,s.px[1]-4,9,9)})}
}
function go(m,i){clearInterval(TM);var s=S[m];s.step=i;s.fresh=1;if(M[m][i]=="Pooling"){s.K=s.Ks[s.f];var o=fwd(m).F[s.f],dd=D[m],bz=-1e9;o.P.forEach(function(r,a){r.forEach(function(v,b){var z=vals(o.B,a*dd.ph,b*dd.pw,dd.ph,dd.pw),sc=(z.some(function(t){return t<=0})&&v>0?1000:0)+v;if(sc>bz){bz=sc;s.p=[a,b]}})})}R(m);window.scrollTo({top:0,behavior:"smooth"})}
function ed(m,k,i,j,v){S[m][k][i][j]=Number(v)||0;R(m)}
function pk(m,k,i,j){S[m][k]=[i,j];R(m)}
function setN(m,v){S[m].n=+v;S[m].W=null;grid(),buildDS(),tgrid();R(m)}
function pickpx(e,c){var r=c.getBoundingClientRect();S.img.px=[Math.min(159,Math.max(0,Math.floor((e.clientX-r.left)/r.width*160))),Math.min(159,Math.max(0,Math.floor((e.clientY-r.top)/r.height*160)))];R("img")}
function up(inp){var fl=inp.files[0];inp.value="";if(!fl)return;rdf(fl).then(function(c){sx.fillStyle="#fff";sx.fillRect(0,0,160,160);sx.drawImage(c,0,0);grid();R("img")})}
function chk(m,sum){var v=$(m+"a").value,e=$(m+"r");if(v===""){e.className="calc ko";e.textContent="Entrez un nombre d’abord.";return}
var okk=Math.abs(Number(v)-sum)<.015;e.className="calc "+(okk?"ok":"ko");e.innerHTML="<b>"+(okk?"Correct.":"Pas tout à fait.")+"</b> Vous avez entré "+v+".<br>Méthode : on multiplie chaque valeur de la zone par la valeur du filtre, puis on additionne tous les produits.<br>Correction : "+S[m].corr}
function play(m,k){clearInterval(TM);var s=S[m],d=D[m],A=conv(s.I,s.K),cols=k=="c"?A[0].length:Math.floor(A[0].length/d.pw),tot=k=="c"?A.length*cols:Math.floor(A.length/d.ph)*cols,i=0;
TM=setInterval(function(){if(i>=tot){clearInterval(TM);return}s[k]=[Math.floor(i/cols),i%cols];i++;R(m)},450)}
function loss(m){var s=S[m],t=conv(s.I,Object.values(D[m].pk)[0]),o=conv(s.I,s.K),e=0,c=0;o.forEach((r,i)=>r.forEach((v,j)=>{e+=Math.pow(v-t[i][j],2);c++}));return e/c}
var lb=m=>`<div class="learn"><h3 style="margin:0 0 4px;font-size:16px;font-weight:500">Comment le filtre est-il choisi ?</h3><p>Il n’est pas choisi à la main. Le concepteur fixe la taille et le nombre de filtres. Leurs valeurs sont d’abord aléatoires, puis apprises pendant l’entraînement : le réseau les corrige pour réduire l’erreur (descente de gradient).</p><p>Essayez : la cible est le filtre « ${Object.keys(D[m].pk)[0]} ». Partez d’un filtre aléatoire et regardez l’erreur baisser.</p><div class="row"><button onclick="rndK('${m}')">Filtre aléatoire</button><button class="go" onclick="learn('${m}',1)">Apprendre ×1</button><button class="go" onclick="learn('${m}',100)">Apprendre ×100</button></div><div class="calc">Erreur actuelle : <b>${f2(loss(m))}</b></div></div>`;
function learn(m,n){var s=S[m],Kt=Object.values(D[m].pk)[0],I=s.I,kh=s.K.length,kw=s.K[0].length,mx2=Math.max(.01,...I.flat().map(v=>v*v)),lr=.4/(kh*kw*mx2);
for(var it=0;it<n;it++){var o=conv(I,s.K),t=conv(I,Kt),G=s.K.map(r=>r.map(()=>0)),N=o.length*o[0].length;
for(var i=0;i<o.length;i++)for(var j=0;j<o[0].length;j++){var e=o[i][j]-t[i][j];for(var a=0;a<kh;a++)for(var b=0;b<kw;b++)G[a][b]+=2*e*I[i+a][j+b]/N}
s.K=s.K.map((r,a)=>r.map((v,b)=>v-lr*G[a][b]))}
s.K=s.K.map(r=>r.map(v=>+v.toFixed(3)));R(m)}
document.addEventListener("keydown",function(e){if(e.target.tagName=="INPUT"||e.target.tagName=="SELECT")return;var s=S[cur];if(e.key=="ArrowRight"&&s.step<M[cur].length-1)go(cur,s.step+1);if(e.key=="ArrowLeft"&&s.step>0)go(cur,s.step-1)});

var CLS={img:["Chat","Chien","Poule"]};
function stk(m){var f=fwd(m);return {L:f.F.map(function(o,i){return {n:"Filtre "+(i+1),P:o.P,K:o.K}}),vec:f.vec}}
function Wm(m,N){var sc=.4;return [0,1,2].map(function(j){return Array.from({length:N},function(_,i){return +(sc*Math.sin((i+1)*(j+2)*1.7+j)).toFixed(3)})})}
function zf(m){var x=stk(m).vec,W=getW(m),b=S[m].b;return W.map(function(w,j){return f2(w.reduce(function(a,wi,i){return a+wi*x[i]},0)+b[j][0])})}
function sm(z){var mx=Math.max.apply(null,z),e=z.map(function(v){return Math.exp(v-mx)}),t=e.reduce(function(a,c){return a+c},0);return e.map(function(v){return v/t})}
var NS=["Convolution","ReLU","Pooling","Empilement","Flatten","Couche dense","Softmax"];
function labOf(m){var s=S[m];return s.lab==null?-1:s.lab}
function rk(m){var d=D[m],k=d.pk[Object.keys(d.pk)[0]];return [0,1,2].map(function(){var a=k.map(function(r){return r.map(function(){return Math.random()-.5})}),fl=a.flat(),mu=fl.reduce(function(x,y){return x+y},0)/fl.length;return a.map(function(r){return r.map(function(v){return f2(v-mu)})})})}
function Wi(m,N){var sc=.03;return [0,1,2].map(function(j){return Array.from({length:N},function(_,i){return +(sc*Math.sin((i+1)*(j+2)*1.7+j)).toFixed(4)})})}
function fwd(m){var s=S[m],d=D[m],F=s.Ks.map(function(K){var A=conv(s.I,K),B=s.relu?A.map(function(r){return r.map(function(v){return Math.max(0,v)})}):A;return {K:K,A:A,B:B,P:pool(B,d.ph,d.pw,s.type)}});return {F:F,vec:[].concat.apply([],F.map(function(o){return o.P.flat()}))}}
function getW(m){var s=S[m],N=fwd(m).vec.length;if(!s.W||s.W[0].length!=N){s.W=Wi(m,N);s.b=[[0],[0],[0]];s.tr=0;s.h=[]}return s.W}
function train(m,n){var s=S[m],d=D[m];getW(m);var y,I,ph=d.ph,pw=d.pw,m2,I0=s.I,B=m=="img"?DSI:[{I:s.I,y:labOf(m)}];if(B[0].y<0)return;
for(var t=0;t<n*B.length;t++){var sp=B[t%B.length];s.I=sp.I;y=sp.y;I=s.I;m2=I.flat().reduce(function(a,v){return a+v*v},0)/I.flat().length;var f=fwd(m),x=f.vec,z=s.W.map(function(w,j){return w.reduce(function(a,wi,i){return a+wi*x[i]},0)+s.b[j][0]}),p=sm(z),dz=p.map(function(v,j){return v-(j==y?1:0)}),
dx=x.map(function(_,i){return s.W.reduce(function(a,w,j){return a+w[i]*dz[j]},0)}),lrW=.5/(1+x.reduce(function(a,v){return a+v*v},0)),lrK=LK*.2/(1+4*m2),off=0;
LS+=-Math.log(Math.max(1e-9,p[y]));if(t%B.length==B.length-1){s.h.push(LS/B.length);LS=0}
f.F.forEach(function(o){var dA=o.A.map(function(r){return r.map(function(){return 0})});
for(var pi=0;pi<o.P.length;pi++)for(var pj=0;pj<o.P[0].length;pj++){var g=dx[off++],cs=[],bi=null;for(var a=0;a<ph;a++)for(var b=0;b<pw;b++){var c=[pi*ph+a,pj*pw+b];cs.push(c);if(!bi||o.B[c[0]][c[1]]>o.B[bi[0]][bi[1]])bi=c}
if(s.type=="max")dA[bi[0]][bi[1]]+=g;else cs.forEach(function(c){dA[c[0]][c[1]]+=g/cs.length})}
if(s.relu)dA=dA.map(function(r,i){return r.map(function(v,j){return o.A[i][j]>0?v:0})});
var dK=o.K.map(function(r){return r.map(function(){return 0})});
for(var i=0;i<dA.length;i++)for(var j=0;j<dA[0].length;j++)for(var a=0;a<o.K.length;a++)for(var b=0;b<o.K[0].length;b++)dK[a][b]+=dA[i][j]*I[i+a][j+b];
o.K.forEach(function(r,a){r.forEach(function(v,b){r[b]=v-lrK*dK[a][b]})})});
s.W.forEach(function(w,j){w.forEach(function(wi,i){w[i]=wi-lrW*dz[j]*x[i]})});s.b.forEach(function(r,j){r[0]-=lrW*dz[j]})}
s.I=I0;s.tr+=n;if(s.h.length>300)s.h=s.h.slice(-300);
s.Ks=s.Ks.map(function(K){return K.map(function(r){return r.map(function(v){return +v.toFixed(3)})})});s.W=s.W.map(function(r){return r.map(function(v){return +v.toFixed(4)})});s.b=s.b.map(function(r){return [+r[0].toFixed(3)]})}
var LS=0;function isDone(m){return acc()[1]<.05||S[m].tr>=300}
function auto(m){clearInterval(TM);TM=setInterval(function(){train(m,3);R(m);if(isDone(m))clearInterval(TM)},100)}
var TRMSG="";function addTr(inp,k){var fs=[].slice.call(inp.files);inp.value="";if(!fs.length)return;TRMSG="";Promise.all(fs.map(rdf)).then(function(cs){var ref=sig16(TSRC),rm=sig16(mir(TSRC)),bad=0;cs.forEach(function(c){var a=sig16(c);if(dst(a,ref)<6||dst(a,rm)<6)bad++;else TR[k].push(c)});if(bad)TRMSG=bad+" photo(s) refusée(s) : identique(s) à la photo de test.";buildDS();rst("img")})}
var LK=1;
function rst(m){var s=S[m];s.Ks=rk(m);s.W=null;s.tr=0;s.h=[];getW(m);R(m)}
function trn(m,n){train(m,n);R(m)}
function stat(m){return ""}
function curve(H,u){if(H.length<2)return '<p>Appuie sur « Entraîner » : la courbe de l’erreur apparaît ici.</p>';var mx=Math.max.apply(null,H)||1,n=H.length,pts=H.map(function(v,i){return (36+i/(n-1)*252).toFixed(1)+","+(120-v/mx*100).toFixed(1)}).join(" ");return '<h3>Courbe de l’erreur</h3><svg viewBox="0 0 300 160" class="cv"><line x1="36" y1="20" x2="36" y2="120" stroke="#000"/><line x1="36" y1="120" x2="290" y2="120" stroke="#000"/><polyline points="'+pts+'" fill="none" stroke="#c2410c" stroke-width="2.5"/><text x="40" y="14" font-size="10" fill="#000">Erreur ↑</text><text x="2" y="30" font-size="10" fill="#000">'+f2(mx)+'</text><text x="24" y="123" font-size="10" fill="#000">0</text><text x="36" y="136" font-size="10" fill="#000">1</text><text x="290" y="136" font-size="10" text-anchor="end" fill="#000">'+n+'</text><text x="163" y="152" font-size="11" text-anchor="middle" fill="#000">'+u+' →</text></svg><p>Chaque point = l’erreur moyenne après un passage. <b>Si la ligne descend, le réseau apprend.</b> Début : <b>'+f2(H[0])+'</b> · maintenant : <b>'+f2(H[n-1])+'</b>.</p>'}
function wb(){var m=cur,s=S[m];return '<div class="learn"><h3>D’où viennent W et b ?</h3><p>Les valeurs de <b>W</b> et de <b>b</b> ne sont pas choisies à la main : elles sont <b>obtenues grâce à l’entraînement</b>. À chaque itération, le réseau les corrige pour réduire l’erreur'+(s.tr?' ('+s.tr+' itérations déjà effectuées)':'')+'.</p><button class="go" onclick="go(\''+m+'\','+M[m].indexOf("Entraînement")+')">Voir l’entraînement →</button></div>'}
function tgrid(){S.img.T=(TSRC&&!TDUP)?G8(tx):null}
function PT(nm){return nm=="Prédiction"?"Partie 2 · Test":nm=="Code complet"?"Code":"Partie 1 · Apprentissage"}
var tc=document.createElement("canvas");tc.width=tc.height=160;var tx=tc.getContext("2d");tx.fillStyle="#fff";tx.fillRect(0,0,160,160);
function tgridOLD(){var s=S.img,n=s.n,d=tx.getImageData(0,0,160,160).data,b=Math.floor(160/n),T=[];for(var i=0;i<n;i++){T.push([]);for(var j=0;j<n;j++){var t=0;for(var y=i*b;y<(i+1)*b;y++)for(var x=j*b;x<(j+1)*b;x++){var k=4*(y*160+x);t+=.299*d[k]+.587*d[k+1]+.114*d[k+2]}T[i].push(f2(Math.round(t/(b*b))/255))}}s.T=T}
function upT(inp){var fl=inp.files[0];inp.value="";if(!fl)return;rdf(fl).then(setTest)}
function esc(t){return t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}
function rl(){var v=Number($("rx").value)||0;$("ro").textContent=Math.max(0,v)}
function showc(m){var e=$(m+"r");e.className="ok";e.textContent="Correction : "+S[m].corr}
function cpy(b){var t=b.parentNode.nextSibling.textContent;try{navigator.clipboard.writeText(t);b.textContent="Copié"}catch(e){b.textContent="Sélectionnez le code puis copiez"}}
var CK={
"Photo":[`img = cv2.imread("chat.jpg")   # image couleur : 3 valeurs (B, G, R) par pixel`],
"Niveaux de gris":[`gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 1 valeur (0 à 255) par pixel`],
"Pixels":[`small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)   # grille 8x8`],
"Normalisation":[`x = small / 255.0   # valeurs entre 0 et 1`],
"Convolution":`layers.Conv2D(filters=3, kernel_size=(3, 3))   # 3 filtres 3x3, valeurs aléatoires au départ`,
"ReLU":`layers.ReLU()   # ou Conv(..., activation="relu")`,
"Pooling":`layers.MaxPooling2D(pool_size=(2, 2))`,
"Empilement":`layers.Conv2D(filters=3, kernel_size=(3, 3))   # filters=3 : 3 filtres, donc 3 cartes empilées`,
"Flatten":`layers.Flatten()   # cartes 3D, vecteur 1D`,
"Couche dense":`layers.Dense(3)   # z = W·x + b : W et b sont créés puis appris automatiquement`,
"Softmax":`layers.Softmax()   # ou Dense(3, activation="softmax")`,
"Entraînement":`# ===== PARTIE 1 : APPRENTISSAGE SUPERVISÉ =====
import glob, numpy as np, cv2
from tensorflow import keras
from tensorflow.keras import layers

CLASSES = ["Chat", "Chien", "Poule"]            # 0, 1, 2

# 1) Données étiquetées : un dossier par classe (travail humain)
#    data/chat/*.jpg   data/chien/*.jpg   data/poule/*.jpg
X, y = [], []
for label, name in enumerate(["chat", "chien", "poule"]):
    for path in glob.glob(f"data/{name}/*.jpg"):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)                 # niveaux de gris
        img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)  # même taille pour toutes
        X.append(img / 255.0)                                        # normalisation : 0 à 1
        y.append(label)                                              # l'étiquette
X = np.array(X).reshape(-1, 8, 8, 1)    # (nb images, 8, 8, 1 canal)
y = np.array(y)

# 2) Architecture du CNN : on la définit UNE seule fois ici
model = keras.Sequential([
    layers.Input(shape=(8, 8, 1)),
    layers.Conv2D(3, (3, 3), activation="relu"),   # convolution + ReLU (3 filtres)
    layers.MaxPooling2D((2, 2)),                   # pooling
    layers.Flatten(),                              # flatten
    layers.Dense(3, activation="softmax"),         # couche dense + softmax (3 classes)
])

# 3) Entraînement : le CNN corrige filtres, W et b pour réduire l'erreur
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=100, validation_split=0.2)  # 80 % apprentissage, 20 % vérification

# 4) Sauvegarde : le modèle garde les filtres, W et b appris
model.save("cnn_chat_chien_poule.keras")`,
"Prédiction":`# ===== PARTIE 2 : TEST (rien n'est redéfini) =====
import numpy as np, cv2
from tensorflow import keras

CLASSES = ["Chat", "Chien", "Poule"]

# 1) On recharge le modèle déjà entraîné : couches + filtres, W et b appris
model = keras.models.load_model("cnn_chat_chien_poule.keras")

# 2) Nouvelle image : MÊME préparation qu'à l'apprentissage
img = cv2.imread("nouvelle_photo.jpg", cv2.IMREAD_GRAYSCALE)  # niveaux de gris
img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)   # même taille (8 x 8)
x = (img / 255.0).reshape(1, 8, 8, 1)                         # normalisation + forme du lot

# 3) Prédiction : le modèle enchaîne seul convolution, ReLU, pooling, flatten, dense, softmax
p = model.predict(x)[0]                       # 3 probabilités
print(CLASSES[int(np.argmax(p))], p)          # classe la plus probable`};
function cd(m,nm){var c=CK[nm];if(!c)return "";if(Array.isArray(c))c=c[0];return c?'<details class="code"'+""+'><summary>Côté programmation'+(nm=="Entraînement"?" · Partie 1 : apprentissage":nm=="Prédiction"?" · Partie 2 : test":" : l’instruction de cette étape")+'</summary><pre>'+esc(c)+"</pre></details>":""}
var FULL={img:`import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers

CLASSES = ["Chat", "Chien", "Poule"]

# 1) Préparer les données (travail humain)
img = cv2.imread("chat.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA) / 255.0
X = img.reshape(1, 8, 8, 1)
y = np.array([0])   # étiquette : 0 = Chat

# 2) Concevoir l'architecture (travail humain)
model = keras.Sequential([
    layers.Input(shape=(8, 8, 1)),
    layers.Conv2D(3, (3, 3), activation="relu"),   # convolution + ReLU (3 filtres)
    layers.MaxPooling2D((2, 2)),                   # pooling
    layers.Flatten(),                              # flatten
    layers.Dense(3, activation="softmax"),         # couche dense + softmax
])

# 3) Entraîner : le CNN apprend filtres, W et b
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(X, y, epochs=100, verbose=0)

# 4) Prédire
p = model.predict(X)
print(CLASSES[int(np.argmax(p))], p)
`};
function rndK(m){S[m].Ks[S[m].f]=rk(m)[0];R(m)}
function pre(m,n){S[m].Ks[S[m].f]=cp(D[m].pk[n]);R(m)}
var PH=null,PHs={},nl=0;
Object.keys(PU).forEach(function(k){var im=new Image();im.onload=function(){PHs[k]=im;if(++nl==Object.keys(PU).length)startImg()};im.src=PU[k]});
function startImg(){PH=PHs.chat;TR[0]=[wc(PHs.chat)];TR[1]=[wc(PHs.chien)];TR[2]=[wc(PHs.poule)];TO=[wc(PHs.o_chat),wc(PHs.o_chien),wc(PHs.o_poule)];
S.img={step:0,n:8,px:[80,80],Ks:rk("img"),f:0,h:[],relu:true,type:"max",c:[0,0],p:[0,0],W:null,tr:0,lab:0,b:[[0],[0],[0]]};drawCat();grid();buildDS();TSRC=wc(PHs.test);tx.drawImage(TSRC,0,0);TDUP=sameAsTrain(TSRC)?1:0;tgrid();cur="img";R("img")}
