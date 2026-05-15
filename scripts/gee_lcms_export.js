// Generated code for assets at top of script
var lcms = ee.ImageCollection("USFS/GTAC/LCMS/v2024-10"),
    damage_template = ee.Image("projects/forest-lst/assets/damage_template"),
    extract_area = 
    /* color: #d63000 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-119.55275392110342, 37.4689477663507],
          [-119.55275392110342, 37.10182584582536],
          [-119.01991700704092, 37.10182584582536],
          [-119.01991700704092, 37.4689477663507]]], null, false);

var start_year = ee.Number(lcms.aggregate_min("year"));
var end_year = ee.Number(lcms.aggregate_max("year"));

print("Start year", start_year);
print("End year", end_year);

var years = ee.List.sequence(start_year.add(1), end_year);

var props = ["year", "study_area", "version"];
var scale = 300; // m
var forest_cutoff = 0.8;

lcms = lcms
  .filter(ee.Filter.eq("study_area", "CONUS"));
  
// Make forest mask
var is_forest_anytime = lcms.map(function(img) {
  return img.select("Land_Cover").lte(5);
}).max().setDefaultProjection({
  crs: lcms.first().projection()
}).reduceResolution({
  reducer: ee.Reducer.mean(),
  maxPixels: 2e4
}).reproject({
  crs: lcms.first().projection(),
  scale: scale
}).gte(forest_cutoff);

Map.addLayer(is_forest_anytime, {}, "forest mask");

// Prepare images for export.
// Get loss fraction, and respective proportion of loss from
// drought/insects and fire specifically.
var lcms_dim = ee.ImageCollection(years.map(function(y) {
  y = ee.Number(y);
  var post = ee.Image(lcms.filter(ee.Filter.eq("year", y)).first());
  var pre = ee.Image(lcms.filter(ee.Filter.eq("year", y.subtract(1))).first());
  
  // Identify transitions from pure forest to mixed forest
  // or from mixed forest to non-forest.
  var pre_forest      = pre.select("Land_Cover").eq(1);
  var pre_mix_forest  = pre.select("Land_Cover").gte(2).and(pre.select("Land_Cover").lte(5));
  
  var post_mix_forest = post.select("Land_Cover").gte(2).and(post.select("Land_Cover").lte(5));
  var post_non_forest = post.select("Land_Cover").gte(6);

  var forest_to_mix = pre_forest.and(post_mix_forest);
  var forest_to_non_forest = pre_forest.and(post_non_forest);
  var mix_to_non_forest = pre_mix_forest.and(post_non_forest);
  var loss_frac = forest_to_mix.or(mix_to_non_forest).or(forest_to_non_forest);
  
  // Identify changes caused by drought-induced mort and fire.
  var dim_change  = post.select("Change").gte(10).and(post.select("Change").lte(12));
  var fire_change = post.select("Change").eq(6).or(post.select("Change").eq(7));
  
  var composite = ee.Image.cat(
    loss_frac.and(dim_change).rename("dim_change"),
    loss_frac.and(fire_change).rename("fire_change"),
    loss_frac.rename("loss_fraction")
  ).copyProperties(post, props);
  
  return composite;
}));


var vis_dim = {
  bands: ["dim_change", "fire_change", "loss_fraction"]
};

Map.addLayer(lcms_dim.filter(ee.Filter.eq("year", 2016)), vis_dim, "2016 change");
Map.addLayer(lcms.filter(ee.Filter.eq("year", 2016)).select("Change"), {}, "LCMS 2016 change");

var lcms_rr = lcms_dim.map(function (img) {
  // Force mean reduction on resolution change.
  // Multiply by 100 and cast to uint8 for
  // better compression.
  var dim_rr = img.reduceResolution({
    reducer: ee.Reducer.mean(),
    maxPixels: 2e4
  }).multiply(
    100
  ).resample(
    "bilinear"
  ).reproject({
    crs: lcms.first().projection(),
    scale: scale
  }).updateMask(
    is_forest_anytime
  ).copyProperties(
    img, props
  ).set(
    "system:time_start", ee.Date.fromYMD(img.get("year"), 1, 1)
  );
  return dim_rr;
});


// test image
var test_dim = lcms_rr.filter(ee.Filter.eq("year", 2016)).first();
print(test_dim.projection());

Map.addLayer(test_dim, vis_dim, "2016 change rr");

/*
//Map.addLayer(lcms_sum, {min: 0, max: lcms.size().getInfo()}, "dim_sum");
Map.addLayer(test_dim, {min: 0, max: 10}, "test_dim");
//Map.addLayer(damage_template.geometry());
Map.addLayer(damage_template.geometry().bounds());
*/


for (var i = start_year.getInfo()+1; i <= end_year.getInfo(); i++) {
  Export.image.toDrive({
      image: lcms_rr.filter(ee.Filter.eq("year", i)).first(),
      region: damage_template.geometry().bounds(),
      folder: "lcms_export_v2",
      description: "dim_" + i,
      shardSize: 64,
      scale: scale,
      crs: "EPSG:5071"
  });
}


