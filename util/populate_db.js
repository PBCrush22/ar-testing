const fs = require('node:fs')
const path = require('node:path')

const base_map_path = path.join(__dirname, '../server/main/base-map')
const base_maps = fs.readdirSync(base_map_path).filter(file => file.endsWith('.geojson'))
for (const file of base_maps){
    const map = JSON.parse(fs.readFileSync(path.join(base_map_path, file), 'utf8'))
    for (const features of map.features){
        fetch(`http://127.0.0.1:5000/localization/poi/post`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({'id': features.properties.id, 'name': features.properties.title, 'lat': features.properties.point.coordinates[0], 'long': features.properties.point.coordinates[1]})
        })
    }
}





