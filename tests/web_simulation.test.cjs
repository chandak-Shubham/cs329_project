// No npm dependencies. Run: node --test tests/web_simulation.test.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const vm = require('node:vm');
const html = readFileSync(join(__dirname,'../index.html'),'utf8');
vm.runInThisContext(html.match(/<script id="simulation-engine">([\s\S]*?)<\/script>/)[1]);
const {SimulationEngine:Engine, NeuralActor, SeededRandom} = globalThis.Kinship;
const data = JSON.parse(html.match(/<script id="research-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const actor = new NeuralActor(data.neural);
function empty(options={}) {return new Engine({initialPop:0,L:10,maxCapacity:10,cBase:0,cDensity:0,mutationRate:0,...options});}
function place(world,x,y,options={}) {return world.spawn(world.cell(x,y),options.genome??123,options);}
function occupancy(world) {
  const agents = world.agents;
  assert.equal(new Set(agents.map(a=>world.cell(a.x,a.y))).size,agents.length);
  assert.equal(world.grid.filter(v=>v!==-1).length,agents.length);
  for(const a of agents) {assert.equal(world.grid[world.cell(a.x,a.y)],a.slot);assert.ok(Number.isFinite(a.energy));assert.ok(a.energy>0);}
  assert.equal(new Set(agents.map(a=>a.id)).size,agents.length);
}

test('Hamilton counts use decision-time relatedness and a strict inequality',()=>{
  const w=empty({B:4,C:2});
  const a=place(w,1,1,{genome:0}),b=place(w,2,1,{genome:0});
  place(w,3,1,{genome:255});w.updateCues();
  w.step([5,0,5]);
  assert.deepEqual(w.donations.map(d=>d.relatedness),[1,0]);
  assert.equal(w.hamiltonAbove,1);assert.equal(w.hamiltonBelow,1);
  w.setParameter('B',2);w.step([5,0,0]);
  assert.equal(w.hamiltonAbove,1);assert.equal(w.hamiltonBelow,2);
  assert.equal(w.hamiltonAbove+w.hamiltonBelow,w.totalDonations);
  assert.equal(w.history.at(-1).hamiltonBelow,2);
  assert.match(w.toCSV(),/hamiltonAbove,hamiltonBelow/);
  w.reset();assert.equal(w.hamiltonAbove,0);assert.equal(w.hamiltonBelow,0);
});

test('donation relatedness survives same-step demographic changes',()=>{
  const w=empty({B:6,C:2,cBase:1});
  place(w,1,1,{genome:0,energy:2.5});place(w,2,1,{genome:0});
  place(w,3,1,{genome:255,energy:0.5});w.updateCues();
  w.step([5,0,0]);
  assert.equal(w.deaths,2);assert.equal(w.agents.length,1);
  assert.equal(w.donations[0].relatedness,1);assert.equal(w.hamiltonAbove,1);
});

test('same seed and interventions reproduce the full trajectory',()=>{
  const a=new Engine({neuralActor:actor}),b=new Engine({neuralActor:actor});
  for(let i=0;i<100;i++){if(i===20){a.injectMimics();b.injectMimics();}if(i===30){a.startShock();b.startShock();}a.step();b.step();}
  assert.deepEqual(a.history,b.history);assert.deepEqual(a.agents,b.agents);
  const other=new Engine({seed:101});assert.notDeepEqual(new Engine().agents,other.agents);
});
test('toroidal neighbors are distance-then-slot sorted and observations cap at 8',()=>{
  const w=empty({maxCapacity:20}),a=place(w,0,0);place(w,9,0);place(w,0,1);
  for(let i=1;i<=8;i++)place(w,i%4+1,2);
  assert.deepEqual(w.neighbors(a).slice(0,2).map(b=>b.slot),[1,2]);
  assert.deepEqual(w.delta(a,w.slots[1]),[-1,0]);
  assert.equal(w.observation(a,w.neighbors(a).slice(0,8)).input.length,43);
});
test('simultaneous swaps, cycles and chains preserve occupancy',()=>{
  const w=empty();const a=place(w,3,3),b=place(w,3,4);
  w.step([1,2]);assert.deepEqual([a.y,b.y],[4,3]);occupancy(w);
  w.resolveMovements(new Map([[a.slot,w.cell(4,4)],[b.slot,w.cell(3,4)]]));occupancy(w);assert.equal(b.y,4);
  const c=place(w,5,4);w.resolveMovements(new Map([[a.slot,w.cell(3,4)],[b.slot,w.cell(5,4)],[c.slot,w.cell(4,4)]]));occupancy(w);
});
test('movement into stationary blockers cancels the entire upstream chain',()=>{
  const w=empty();place(w,1,1);place(w,2,1);place(w,3,1);
  w.step([3,3,0]);assert.deepEqual(w.agents.map(a=>a.x),[1,2,3]);occupancy(w);
});
test('competing movement destinations admit exactly one winner',()=>{
  const w=empty();place(w,1,1);place(w,3,1);w.step([3,4]);
  assert.equal(w.agents.filter(a=>a.x===2).length,1);occupancy(w);
});
test('donations use pre-movement target identities and post-movement crowding',()=>{
  const w=empty({cBase:0.5,cDensity:0.1});const a=place(w,1,1),b=place(w,4,1);
  w.step([5,3]);assert.equal(b.x,5);assert.equal(a.energy,27.5);assert.equal(b.energy,35.5);assert.equal(w.lastDonations,1);
  assert.equal(w.donations[0].recipientId,b.id);assert.equal(w.reputation(a),1);
});
test('donation affordability is strict and mimics cannot donate',()=>{
  const w=empty();const a=place(w,1,1,{energy:2}),b=place(w,2,1,{mimic:true});
  w.step([5,5]);assert.equal(w.lastDonations,0);assert.equal(a.energy,2);assert.equal(b.energy,30);
  a.energy=2.1;w.step([5,5]);assert.equal(w.lastDonations,1);assert.ok(Math.abs(a.energy-0.1)<1e-9);
});
test('mitosis conserves energy, inherits identity and resets reputation',()=>{
  const w=empty();const a=place(w,4,4,{energy:80,history:[1,1],mimic:true});w.step([0]);
  assert.equal(w.agents.length,2);assert.equal(w.agents.reduce((sum,a)=>sum+a.energy,0),80);
  const b=w.agents.find(b=>b.id!==a.id);assert.equal(b.parentId,a.id);assert.equal(b.energy,40);assert.equal(b.genome,a.genome);assert.equal(b.mimic,true);assert.equal(b.age,0);assert.equal(w.reputation(b),0.5);occupancy(w);
});
test('same-step starvation and slot reuse retain distinct individual identities',()=>{
  const w=empty({cBase:1});const dead=place(w,1,1,{energy:0.5}),parent=place(w,4,4,{energy:80});w.step([0,0]);
  assert.equal(w.deathEvents[0].id,dead.id);assert.equal(w.slots[dead.slot].parentId,parent.id);assert.notEqual(w.slots[dead.slot].id,dead.id);assert.equal(w.slots[dead.slot].age,0);occupancy(w);
});
test('mutation probability one flips every bit; newborns cannot reproduce immediately',()=>{
  const w=empty({mutationRate:1});const a=place(w,4,4,{energy:200,genome:170});w.step([0]);
  assert.equal(w.agents.length,2);assert.equal(w.agents[1].genome,85);assert.equal(a.energy,100);
});
test('all policy decisions use the same prior reputation snapshot',()=>{
  const w=empty({strategy:'reputation'});place(w,4,4);place(w,4,5);w.updateCues();w.step();assert.equal(w.lastDonations,2);
  assert.equal(w.agents[0].energy,34);assert.equal(w.agents[1].energy,34);
});
test('history is a rolling 10-action fraction and not a persistent prior',()=>{
  const w=empty();const a=place(w,4,4);place(w,4,5);assert.equal(w.reputation(a),0.5);
  w.step([5,0]);assert.equal(w.reputation(a),1);
  for(let i=0;i<10;i++)w.step([0,0]);assert.equal(w.reputation(a),0);assert.equal(a.history.length,10);
});
test('relatedness excludes self pairs and uses living-population baseline',()=>{
  const w=empty();const a=place(w,1,1,{genome:0}),b=place(w,2,1,{genome:0}),c=place(w,3,1,{genome:255});w.updateCues();
  assert.equal(w.hBar,1/3);assert.equal(w.relatedness(a,b),1);assert.equal(w.relatedness(a,c),0);assert.equal(w.relatedness(a,a),0);
  c.genome=0;w.updateCues();assert.equal(w.relatedness(a,b),0);
});
test('mimics copy dominant resident tag, use only free slots and never overwrite',()=>{
  const w=empty({maxCapacity:5});place(w,4,4,{genome:5});place(w,4,5,{genome:7});place(w,4,6,{genome:7});w.updateCues();
  assert.equal(w.injectMimics(12),2);assert.equal(w.agents.filter(a=>a.mimic&&a.genome===7).length,2);assert.equal(w.injectMimics(12),0);occupancy(w);
  assert.equal(empty().injectMimics(),0);
});
test('shock lasts exactly 30 steps and restores the latest base drain',()=>{
  const w=empty({cBase:1,eMitosis:1000,maxCapacity:1});const a=place(w,4,4,{energy:300});assert.equal(w.startShock(),true);assert.equal(w.startShock(),false);
  for(let i=0;i<15;i++)w.step([0]);assert.equal(a.energy,270);w.setParameter('cBase',0.5);
  for(let i=0;i<15;i++)w.step([0]);assert.equal(a.energy,255);assert.equal(w.shockRemaining,0);w.step([0]);assert.equal(a.energy,254.5);
});
test('automatic invasion occurs before transition 41 and is represented at state 40',()=>{
  const w=empty({autoInvasion:true,maxCapacity:20});place(w,4,4);for(let i=0;i<40;i++)w.step([0]);assert.equal(w.totalInjected,0);
  w.step([0]);assert.equal(w.totalInjected,12);assert.equal(w.events.find(e=>e.type==='injection').step,40);assert.equal(w.history.find(r=>r.step===40).mimics,12);
  w.step([0]);assert.equal(w.totalInjected,12);
});
test('zero benefit stays finite; extinction and time limits stop progression',()=>{
  const w=empty({cBase:1});place(w,4,4,{energy:1});w.setParameter('B',0);w.step([0]);assert.equal(w.finished,true);assert.equal(w.step(),false);assert.equal(w.metrics().meanEnergy,0);
  const t=empty({maxSteps:2});place(t,4,4);t.step([0]);t.step([0]);assert.equal(t.step(),false);assert.equal(t.stepCount,2);
});
test('CSV records parameter changes and interventions without duplicate state rows',()=>{
  const w=new Engine();w.step();w.injectMimics();w.setParameter('B',1);w.startShock();
  const csv=w.toCSV();assert.equal(w.history.length,2);assert.equal(csv.split('\r\n').length,3);assert.match(csv,/Injected/);assert.match(csv,/B changed to 1/);assert.match(csv,/Base metabolism doubled/);assert.match(csv,/seed,gridSize/);
});
test('neural action sampling respects masks and matches seed replay',()=>{
  const input=Array(43).fill(0.2),mask=Array(13).fill(false);mask[4]=true;
  assert.equal(actor.sample(input,mask,new SeededRandom(3)),4);
  const a=new Engine({strategy:'neural',neuralActor:actor}),b=new Engine({strategy:'neural',neuralActor:actor});
  for(let i=0;i<50;i++){a.step();b.step();occupancy(a);}assert.deepEqual(a.history,b.history);
  const w=empty({strategy:'neural',neuralActor:actor});const mimic=place(w,4,4,{mimic:true});place(w,4,5);
  const sampled=Array.from({length:30},()=>w.chooseAction(mimic));assert.ok(sampled.every(action=>action<5));assert.ok(sampled.some(action=>action>0));
  w.strategy='selfish';assert.ok(Array.from({length:30},()=>w.chooseAction(mimic)).some(action=>action>0));
});
test('all five strategies preserve ecological invariants over multi-seed runs',()=>{
  for(const strategy of ['reputation','fixed','kin','selfish','neural'])for(const seed of [0,42,101]){
    const w=new Engine({strategy,seed,neuralActor:actor,autoInvasion:true,maxSteps:150});
    while(!w.finished){w.step();occupancy(w);assert.ok(w.agents.length<=60);}
    assert.equal(w.agents.length,w.initialPop+w.births+w.totalInjected-w.deaths);
  }
});
test('genome-to-RGB mapping matches the 3/3/2 bit definition',()=>{
  assert.equal(Engine.color(0),'rgb(0,0,0)');assert.equal(Engine.color(255),'rgb(255,255,255)');assert.equal(Engine.color(224),'rgb(255,0,0)');assert.equal(Engine.color(28),'rgb(0,255,0)');assert.equal(Engine.color(3),'rgb(0,0,255)');
});
