import polars as pl
import h3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# データフレームの読み込み
def make_df(DATA_PATH):
    lf_train = pl.scan_csv(DATA_PATH / 'train.csv', infer_schema_length=1000000)
    lf_test = pl.scan_csv(DATA_PATH / 'test.csv', infer_schema_length=1000000)

    train_col = 'money_room'
    test_col = 'index'

    use_col = [
        'target_ym',
        'building_id',
        'building_status',
        'building_type',
        'building_name',
        'unit_count',
        'full_address',
        'building_structure',
        'floor_count',
        'basement_floor_count',
        'land_youto',
        'land_toshi',
        'land_road_cond',
        'land_seigen',
        'building_area_kind',
        'management_form',
        'management_association_flg',
        'room_floor',
        'balcony_area',
        'dwelling_unit_window_angle',
        'snapshot_create_date',
        'new_date',
        'snapshot_modify_date',
        'timelimit_date',
        'flg_open',
        'flg_own',
        'bukken_type',
        'empty_number',
        'empty_contents',
        'walk_distance1',
        'flg_new',
        'house_kanrinin',
        'money_kyoueki',
        'money_sonota_str1',
        'money_sonota1',
        'money_sonota_str2',
        'money_sonota2',
        'money_sonota_str3',
        'money_sonota3',
        'parking_money',
        'parking_money_tax',
        'parking_kubun',
        'parking_distance',
        'parking_memo',
        'genkyo_code',
        'usable_status',
        'usable_date',
        'school_ele_distance',
        'school_jun_distance',
        'convenience_distance',
        'super_distance',
        'hospital_distance',
        'park_distance',
        'drugstore_distance',
        'bank_distance',
        'shopping_street_distance',
        'free_rent_duration',
        'free_rent_gen_timing',

        #前処理用。後でexclude
        'year_built',
        'lon',
        'lat',
        'addr1_1',
        'addr1_2',
        'unit_area',
        'house_area',
        'floor_plan_code',
        'reform_exterior',
        'reform_exterior_other',
        'reform_exterior_date',
        'reform_common_area',
        'reform_common_area_date',
        'reform_date',
        'reform_place',
        'reform_place_other',
        'reform_wet_area',
        'reform_wet_area_other',
        'reform_wet_area_date',
        'reform_interior',
        'reform_interior_other',
        'reform_interior_date',
        'reform_etc',
        'renovation_date',
        'renovation_etc',
        'unit_tag_id',
        'building_tag_id',
        'statuses',
    ]
    
    train_df = (
        lf_train
        .select(
            pl.col(train_col),
            pl.col(use_col)
        )
        .collect(streaming=True)
    )

    test_df = (
        lf_test
        .select(
            pl.col(test_col),
            pl.col(use_col)
        )
        .collect(streaming=True)
    )

    return train_df, test_df



#trainの目的変数'money_room'の外れ値を処理する関数
def train_outlier_correct(input_df:pl.DataFrame) -> pl.DataFrame:
    output_df = input_df.clone()
    output_df = output_df.filter(
        pl.col('money_room') < 3500000
    )
    
    list_money_room_outliner = ['a444899','a367920','a365166','a072280','a407859','a187763', 'a331352', 'a100147']
    
    output_df = (
        output_df
        # 10倍されてしまっていると推測されるものを補正
        .with_columns(
            pl.when(pl.col('building_id').is_in(list_money_room_outliner) & (pl.col('money_room') > 400000))
            .then((pl.col('money_room') / 10).cast(pl.Int64))
            .otherwise(pl.col('money_room'))
        )
        .with_columns(
            pl.when(pl.col('building_id') == 'a145150')
            .then((pl.col('money_room') / 10).cast(pl.Int64))
            .otherwise(pl.col('money_room'))
        )
        # 先頭に1が誤ってついていると思われるものを補正
        .with_columns(
            pl.when((pl.col('building_id') == 'a486200') & (pl.col('money_room') == 161000))
            .then(pl.col('money_room') - 100000)
            .otherwise(pl.col('money_room'))
        )
        # 原因がわからない外れ値はdrop
        .filter(~((pl.col('building_id') == 'a015764') & (pl.col('money_room') == 280800)))
        .filter(pl.col('building_id') != 'a061236')
        
        # 桁が足りないものは10倍
        .with_columns(
            pl.when(((pl.col('building_id') == 'a161233') & (pl.col('money_room') == 7400)) | (pl.col('building_id') == 'a293831'))
            .then(pl.col('money_room') * 10)
            .otherwise(pl.col('money_room'))
            )
    )
    return output_df


#train/testの外れ値を補正する関数
def outliner_treatment(input_df: pl.DataFrame) -> pl.DataFrame:
    output_df = input_df.clone()
    
    # house_areaとunit_areaのうち間取りの中央値に近い値を採用したroom_areaカラムを作成
    output_df = (
        output_df
        # 間取りごとの面積中央値を計算
        .with_columns(pl.median('unit_area').over('floor_plan_code').alias('unit_area_median'))
        .with_columns(
        # nullでない方を採用
        pl.when(pl.col("house_area").is_null())
        .then(pl.col("unit_area"))
        .when(pl.col("unit_area").is_null())
        .then(pl.col("house_area"))
        # 部屋の間取り平均に近いほうを採用
        .when((pl.col('unit_area') - pl.col('unit_area_median')).abs() <=  (pl.col('house_area') - pl.col('unit_area_median')).abs())
        .then(pl.col('unit_area'))
        .when((pl.col('unit_area') - pl.col('unit_area_median')).abs() >  (pl.col('house_area') - pl.col('unit_area_median')).abs())
        .then(pl.col('house_area'))
        # 異常値発見用
        .otherwise(pl.lit(-1))
        .alias('room_area')
        )
        # 外れ値の処理 1000以上なら100で、400以上なら10で割る
        .with_columns(
            pl.when(pl.col('room_area') > 1000)
                .then(pl.col('room_area') / 100)
                .when(pl.col('room_area') > 400)
                .then(pl.col('room_area') /10)
                .otherwise(pl.col('room_area'))
            )
        # 小数点第1位で丸める
        .with_columns(((pl.col('room_area') * 10).round()) / 10)
        # 不要なカラムは除外
        .select(pl.all().exclude('unit_area_median'))
    )
    
    # management_association_flgの外れ値を修正
    output_df =(
        output_df
        .with_columns(
            pl.when((pl.col('management_association_flg') == 1) | (pl.col('management_association_flg') == 2))
            .then(pl.col('management_association_flg'))
        )
    )
    # house_kanrininの外れ値を補正
    output_df = (
        output_df
        .with_columns(
            pl.when(pl.col('house_kanrinin').is_in([1, 2, 3, 4, 5]))
            .then(pl.col('house_kanrinin'))
        )
    )
    
    # floor_countの外れ値を補正(東京都以外の60階以上の建物をnullに)
    output_df = (
        output_df
        .with_columns(
            pl.when(~((pl.col('floor_count') >= 60) & (pl.col('addr1_1') != 13)))
            .then(pl.col('floor_count'))
        )
        #20階以上で部屋数より階数が多いのは外れ値と推測
        .with_columns(
            pl.when(~((pl.col('floor_count') > 20) & (pl.col('floor_count') >= pl.col('unit_count'))))
            .then(pl.col('floor_count'))
        )
        # 0階は外れ値
        .with_columns(
            pl.when(pl.col('floor_count') != 0)
            .then(pl.col('floor_count'))
        )
    )
    
    # basement_floor_countの外れ値を補正
    output_df = (
        output_df
        .with_columns(
            # 地下5階以上の物件はなさそう
            pl.when(pl.col('basement_floor_count')  < 5)
            .then(pl.col('basement_floor_count'))
        )
        .with_columns(
            # 10階建て未満で地下2階以上の物件もないだろう
            pl.when(~((pl.col('basement_floor_count') >= 2) & (pl.col('floor_count') <= 10)))
            .then(pl.col('basement_floor_count'))
        )
        # 0階はnullに置き換え
        .with_columns(
            pl.when(pl.col('basement_floor_count') != 0)
            .then(pl.col('basement_floor_count'))
        )
    )
    
    # 'bukken_type'を二値変数(賃貸マンション:0、賃貸アパート：1)に変換
    output_df = output_df.with_columns(pl.col('bukken_type') - 3101)
    
    # 'management_association_flg'を二値変数(なし：0,あり：1)に変換
    output_df = output_df.with_columns(pl.col('management_association_flg') - 1)
    
    # 間取りがあてにならないのでfloor_plan_codeから計算
    output_df = (
        output_df
        .with_columns(
        (pl.col('floor_plan_code') // 100).alias('room_count_from_floor_plan_code'),
        (pl.col('floor_plan_code') % 100).alias('madori_from_floor_plan_code'),
        )
        .select(
            pl.all().exclude('room_count')
        )
    )
    
    # 共益費が1m^2あたり1500円以上は外れ値と推測
    output_df = (
        output_df
        # 1m^2あたりの共益費を算出
        .with_columns((pl.col('money_kyoueki') / pl.col('room_area')).alias('kyouekihi_per_1m2'))
        .with_columns(
            # 1万円/1m^2以上は100で割る
            pl.when(pl.col('kyouekihi_per_1m2') >= 10000)
            .then((pl.col('money_kyoueki') / 100).cast(pl.Int64))
            # 1500/1m^2以上は10で割る
            .when(pl.col('kyouekihi_per_1m2') >= 1500)
            .then((pl.col('money_kyoueki') / 10).cast(pl.Int64))
            .otherwise(pl.col('money_kyoueki'))
        )
        .select(pl.all().exclude('kyouekihi_per_1m2'))
    )
    
    # フリーレント期間7カ月以上は外れ値と推測
    output_df = (
        output_df
        .with_columns(
            pl.when(pl.col('free_rent_duration') <= 6)
            .then(pl.col('free_rent_duration'))
        )
    )
    
    # 'usable_status'の外れ値をnullに
    output_df = (
        output_df
        .with_columns(
            pl.when(pl.col('usable_status').is_in([1, 2, 3]))
            .then(pl.col('usable_status'))
        )
    )
    
    return output_df

# trainデータの単位面積当たりの価格を算出する関数。
# train_outlier_correct()とoutliner_treatment()のあとに適用する
def make_money_per_1m2(input_df:pl.DataFrame) -> pl.DataFrame:
    output_df = input_df.clone()
    
    output_df = (
        output_df
        .with_columns(
            (pl.col('money_room') / pl.col('room_area')).cast(pl.Int64).alias('money_per_1m2')
        )
    )
    return output_df


# データ取得年('target_year')/データ取得月('target_month')/建築年('year_built')/建築月('month_built')を作成
# 築年数('buildings_age')を算出
def calc_buildings_age(input_df:pl.DataFrame) -> pl.DataFrame:
    output_df = input_df.clone()
    
    output_df = (
        output_df
        # target_ymと建築年月を型変換
        .with_columns(
        pl.col('target_ym').cast(pl.String).str.to_date('%Y%m'),
        pl.col('year_built').cast(pl.String).str.to_date('%Y%m'),
        )
        # 年と月に分ける
        .with_columns(
            pl.col('target_ym').dt.year().alias('target_year'),
            pl.col('target_ym').dt.month().alias('target_month'),
            pl.col('year_built').dt.month().alias('month_built'),
            pl.col('year_built').dt.year()
        )
        # 築年数を表す'buildings_age'を追加
        .with_columns(
            (((pl.col('target_year') * 12 + pl.col('target_month'))
            - (pl.col('year_built') * 12 + pl.col('month_built')))
            // 12)
            .alias('buildings_age')
        )
        # 指定期間以外は外れ値としてnullで埋める
        .with_columns(
        pl.when(pl.col('buildings_age').is_between( -1, 60, closed='both'))
        .then(pl.col('buildings_age'))
        )
    )
    return output_df


def join_money_room_over_building(train_input:pl.DataFrame, test_input:pl.DataFrame)->pl.DataFrame:
    """ビルIDごとの賃料、1平米あたり賃料を計算し、テストデータに結合する関数
    
    Args:
        train_input(pl.DataFrame): 前処理適用後の学習データ
        test_input(pl.DataFrame): 前処理適用後のテストデータ
    Retuens:
        pl.DataFrame: 計算したビルIDごとの賃料が入ったテストデータ
    """
    #
    train_building_money_df = (
        train_input
        .select(['building_id','money_room', 'money_per_1m2'])
        #ビルIDごとの平均値を計算
        .with_columns(
            pl.col('money_room').mean().over('building_id').cast(pl.Int64).alias('money_room_mean_building'),
            pl.col('money_per_1m2').mean().over('building_id').alias('money_per_1m2_mean_building'),
        )
        .sort('building_id')
        .select(pl.all().exclude(['money_per_1m2', 'money_room']))
        .unique()
    )
        #テストデータに結合
    output_df = (
        test_input
        .join(
            train_building_money_df, on='building_id', how='left'
        )
    )
    return output_df


# H3インデックスカラムを追加する処理
def get_h3_col(input_df:pl.DataFrame, resolution:int)->pl.DataFrame:
    output_df = input_df.clone()
    
    # H3インデックスを取得する関数 (緯度, 経度の順で渡す)
    def get_h3_index_res(lat_lon, resolution=resolution):
        lat = lat_lon['lat']
        lon = lat_lon['lon']
        return h3.latlng_to_cell(lat, lon, resolution)
    
    output_df = (
        output_df
        .with_columns(pl.struct(['lat', 'lon']).alias('lat_lon'))
        .with_columns(
            pl.col('lat_lon')
            .map_elements(get_h3_index_res, return_dtype=pl.String)
            .alias(f'h3_index_res{resolution}')
        )
        .select(pl.all().exclude('lat_lon'))
    )
    return output_df

# 'building_tag_id', 'unit_tag_id', 'statuses'を展開する関数
def tag_expand(input_df:pl.DataFrame) -> pl.DataFrame:
    output_df = input_df
    
    tag_master = pl.read_excel('/workspace/data/data_definition.xlsx', sheet_name='③タグマスタ情報')
    tag_master = (
        tag_master
        .with_columns(
            (pl.col('タグ内容') + pl.col('タグID').cast(pl.String)).alias('tag_name_id'),
            pl.col('タグID').alias('tag_id'),
        )
        .select(['tag_id', 'tag_name_id', '補足'])
    )
    
    tag_dict_list = tag_master.to_dicts()
    
    for dict in tag_dict_list:
        id = dict.get('tag_id')
        tag = dict.get('tag_name_id')
        side_note = dict.get('補足')
        
        if side_note == '棟に紐づくもの':
            output_df = (
                output_df
                .with_columns(
                    pl.when(pl.col('building_tag_id').str.contains(id))
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                    .alias(tag)
                )
            )
        elif side_note == '戸に紐づくもの':
            output_df = (
                output_df
                .with_columns(
                    pl.when(pl.col('unit_tag_id').str.contains(id))
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                    .alias(tag)
                )
            )
        else:
            output_df = (
                output_df
                .with_columns(
                    pl.when(pl.col('statuses').str.contains(id))
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                    .alias(tag)
                )
            )
    
    zero_col = [
        'マイホーム発電システム213201',
        '駐車場完備320701',
        '託児所付き323101',
        'タワーマンション330301',
        '3階建て333201',
        '総戸数・総区画数30以上333401',
        '総戸数・総区画数が10以上〜30未満333402',
        'リゾートマンション334301',
        'マンション管理評価書付き393401',
        '浴室1.6×1.8M以上220902',
        'コンロ三口以上230302',
        '高効率給湯器230802',
        '屋根裏収納253101',
        'シューズクローク253201',
        'ブロードバンド260601',
        'ブロードバンド263301',
        'リノベーション物件270101',
        'リフォーム物件273101',
        'LDK15帖以上283101',
        '4LDK以上283301',
        'メゾネット293201',
        '庭が広い293701',
        '屋上・ルーフバルコニー294101',
        'ホームセキュリティ対応313301',
        '100m2以上334401',
        '地下340103',
        '南向き343101',
        '所有権120301',
        'フラット35適用可能123301',
        '瑕疵担保付き123401',
        '仲介手数料不要123501',
        '売主・代理123601',
        '現地内覧可能190101',
        'モデルルーム公開中193201',
        '販売中193301',
        '販売予定193401',
        '自由設計対応193501',
        '間取り変更可能193601',
        '駐車場2台以上320601',
        '保証付住宅330701',
        '住宅性能評価書330901',
        'フラット35・S適合証明書335001',
        '長期優良住宅認定通知書335101',
        '建設住宅性能評価書（新築時）335301',
        '建設住宅性能評価書（既存住宅）335401',
        '法適合状況調査報告書335601',
        '低炭素住宅335701',
        '瑕疵保険（国交省指定）による保証利用可335901',
        '瑕疵保険（国交省指定）による保証付335902',
        '瑕疵保証（不動産会社独自）付336001',
        'インスペクション（建物検査）報告書336101',
        '新築時・増改築時の設計図書336201',
        '修繕・点検の記録336301',
        '更地350101',
        '古家あり350102',
        '即入居可350301',
        '大規模分譲地513601',
        "LIFULL HOME'S 認定物件513701",
        "LIFULL HOME'S 住宅評価513801",
        '瑕疵保険(保証付き)513901',
        '瑕疵保険(利用可能)513902',
        '瑕疵保険(対象外)513903',
        '設備保証付き514001',
        '認定基準以上の設備保証付き514002',
        'シロアリ検査(保証付き)514101',
        'シロアリ検査(不適合又は不明)514102',
        'シロアリ検査(検査合格)514103',
        '価格査定マニュアル活用514201',
        'オンライン内見可514301',
        'オンライン相談可514401',
        'IT重説可514501']
    
    output_df = (
        output_df
        .select(
            pl.all()
            .exclude(zero_col)
            .exclude('building_tag_id', 'unit_tag_id', 'statuses')
            )
        )
    
    return output_df



def make_station_df(open_data_path):
    station_df = gpd.read_file(open_data_path)
    
    station_col = [
        'S12_001',
        'S12_001c',
        'S12_001g', 
        'S12_002',
        'S12_003',
        'S12_034',
        'S12_035', 
        'S12_036', 
        'S12_037', 
        'S12_038', 
        'S12_039', 
        'S12_040',
        'S12_041',
        'S12_042',
        'S12_043',
        'S12_044', 
        'S12_045', 
        'S12_046',
        'S12_047', 
        'S12_048', 
        'S12_049', 
        'S12_050', 
        'S12_051', 
        'S12_052',
        'S12_053', 
        'geometry'
    ]
    # 必要なカラムに絞り込み
    station_df = station_df[station_col]
    
    station_col_rename = {
        'S12_001' :'station_name', 
        'S12_001c':'station_code',
        'S12_001g':'station_group', 
        'S12_002':'operate',
        'S12_003':'route_name',
        'S12_034':'dup2018',
        'S12_035':'data2018', 
        'S12_036':'remarks2018', 
        'S12_037':'count2018', 
        'S12_038':'dup2019', 
        'S12_039':'data2019', 
        'S12_040':'remarks2019',
        'S12_041':'count2019',
        'S12_042':'dup2020',
        'S12_043':'data2020',
        'S12_044':'remarks2020', 
        'S12_045':'count2020', 
        'S12_046':'dup2021',
        'S12_047':'data2021', 
        'S12_048':'remarks2021', 
        'S12_049':'count2021', 
        'S12_050':'dup2022', 
        'S12_051':'data2022', 
        'S12_052':'remarks2022',
        'S12_053':'count2022', 
        'geometry':'geometry'
    }
    # 扱いやすいようにリネーム
    station_df = station_df.rename(columns=station_col_rename)
    
    # 欠損値を0埋めしてint型に変換
    station_df['count2022'] = station_df['count2022'].fillna(0).astype('Int64')
    
    return station_df


def get_station_cols(station_df, input_df):
    # 地図用に抽出
    station_geo = station_df[['station_name', 'station_code', 'station_group', 'operate', 'route_name', 'geometry']]
    
    # 駅の中央でポイントに変換
    station_geo['geometry'] = station_geo['geometry'].apply(lambda line: line.interpolate(0.5, normalized=True))
    
    # x座標とy座標を抽出
    station_geo['x'] = station_geo.geometry.x
    station_geo['y'] = station_geo.geometry.y

    # station_groupごとにグループ化し、xとyの平均を計算
    grouped_coords = station_geo.groupby('station_group').agg({
        'x': 'mean',
        'y': 'mean'
    }).reset_index()

    # 各グループの最初のstation_nameを取得
    first_station_name = station_geo.groupby('station_group')['station_name'].first().reset_index()

    # 平均座標とstation_nameを結合
    result = grouped_coords.merge(first_station_name, on='station_group')

    # 新しいgeometry列を作成
    result['geometry'] = gpd.points_from_xy(result['x'], result['y'])

    # 必要なカラムを選択
    station_group_geo = result[['station_name', 'station_group', 'geometry']]

    # GeoDataFrameに変換
    station_group_geo = gpd.GeoDataFrame(station_group_geo, geometry='geometry')

    # 元のデータフレームと同じCRSを設定
    station_group_geo.crs = station_geo.crs
    
    # 駅から2.4kmの範囲のポリゴンを取得
    station_group_geo = station_group_geo.to_crs('EPSG:32654')
    radius_km = 2.4
    station_group_geo['buffer'] = station_group_geo['geometry'].buffer(radius_km * 1000)
    station_group_geo = station_group_geo.to_crs('EPSG:4326')
    station_group_geo['buffer'] = station_group_geo['buffer'].to_crs("EPSG:4326")
    
    station_polygon = station_group_geo[['station_name', 'station_group', 'buffer']]
    station_polygon = station_polygon.rename(columns={'buffer':'geometry'})
    
    # input_dfの必要なカラムに絞り込み
    input_station = (
        input_df
        .select('lon', 'lat', 'building_id')
        .unique()
        .to_pandas()
    )
    
    # input_dfをGeoDataFrameのデータフレームに変換
    input_station['geometry'] = input_station.apply(lambda row:Point(row['lon'], row['lat']), axis=1)
    input_station_gdf = gpd.GeoDataFrame(input_station, geometry='geometry', crs=station_polygon.crs)
    
    # ポリゴンとポイントが重なるものに絞り込み、結合
    joined = gpd.sjoin(input_station_gdf, station_polygon, how='left', predicate='within')
    station_joined = pd.merge(joined, station_group_geo[['station_group', 'geometry']].rename(columns={'geometry':'station_geometry'}), how='left', on='station_group')
    
    # 距離計算を行う
    station_joined['geometry'] = station_joined['geometry'].to_crs(epsg=3857)
    station_joined['station_geometry'] = station_joined['station_geometry'].to_crs(epsg=3857)
    station_joined['distance_from_station'] = station_joined['geometry'].distance(station_joined['station_geometry'])
    
    station_joined = station_joined[['building_id', 'station_name', 'station_group', 'distance_from_station']]
    station_joined['distance_from_station'] = station_joined['distance_from_station'].dropna().round().astype('Int64')
    
    ## 各物件近い3駅に絞り込み
    #  station_joinedをbuilding_idとdistance_from_stationでソート
    station_sorted = station_joined.sort_values(['building_id', 'distance_from_station'])

    #  各building_idごとに距離の順位をつける
    station_sorted['rank'] = station_sorted.groupby('building_id').cumcount() + 1

    # 各building_idについて距離が小さい上位3つを抽出
    top3_stations = station_sorted[station_sorted['rank'] <= 3]

    # データをピボットして、各順位をカラムに展開
    station_distance = top3_stations.pivot(index='building_id', columns='rank', values=['station_group', 'station_name', 'distance_from_station'])

    # カラムのMultiIndexをフラット化
    station_distance.columns = [f"{col}_{int(rank)}" for col, rank in station_distance.columns]

    # インデックスをリセット
    station_distance.reset_index(inplace=True)
    
    ## 徒歩20分以内の駅数を表すカラム'num_of_stations_available'を作成
    result = joined.groupby('building_id')['station_group'].apply(lambda x: list(x.dropna())).reset_index()
    result['num_of_stations_available'] = result['station_group'].apply(len)
    num_of_stations_available = result[['building_id', 'num_of_stations_available']]
    
    # 1つのデータフレームに集約
    output_df = station_distance.merge(num_of_stations_available, on='building_id', how='left')
    
    # PolarsDataFrameに変換し元のデータフレームと結合
    output_df = pl.from_pandas(output_df)
    output_df = input_df.join(output_df, on='building_id', how='left')
    
    return output_df

def calc_building_id_mean(train_df, test_df, target_col):
    # ビルIDごとの平均を算出(自身を除く)
    train_df = (
        train_df
        .with_columns(
            pl.sum(target_col).over('building_id').alias('money_room_sum_building_id'),
            pl.count(target_col).over('building_id').alias('money_room_count_building_id')
        )
        .with_columns(
            pl.when(pl.col('money_room_count_building_id') >= 2)
            .then(((pl.col('money_room_sum_building_id') - pl.col(target_col)) / (pl.col('money_room_count_building_id') - 1))
                .alias('money_room_mean_building_id'))
        )
        .select(pl.all().exclude('money_room_sum_building_id', 'money_room_count_building_id'))
    )

    # ビルID平均をtestに追加
    money_room_mean_for_test = (
        train_df
        .select('building_id', target_col)
        .with_columns(
            pl.mean(target_col).over('building_id').alias('money_room_mean_building_id')
        )
        .select(pl.col('building_id', 'money_room_mean_building_id'))
        .unique()
    )

    test_df = test_df.join(money_room_mean_for_test, how='left', on='building_id')
    
    return train_df, test_df

def get_station_weight_num(input_df:pl.DataFrame, station_df):
    output_df = input_df.clone()
    station_df = pl.from_pandas(station_df[['station_group', 'count2022']])

    #station_dfのstation_group内の最大値を採用
    station_df = (
        station_df
        .with_columns(
            pl.max('count2022').over('station_group').alias('count2022')
        )
        .unique()
    )
    #最小限のカラムを選択
    tmp = (
            output_df
            .select(
                'building_id',
                'station_group_1',
                'station_name_1',
                'distance_from_station_1',
                'station_group_2',
                'station_name_2',
                'distance_from_station_2',
                'station_group_3',
                'station_name_3',
                'distance_from_station_3',
                )
            .filter(pl.col('station_group_1').is_not_null())
            .unique()
        )
    #近傍3駅分を取得
    for i in range(1, 4):
        tmp = (
            tmp
            .join(station_df,left_on=f'station_group_{i}', right_on='station_group', how='left')
            .rename({'count2022': f'sg{i}_count2022'})
        )

    # 非線形な重み付けをする関数のパラメータ設定
    k = 0.35
    c = 18

    # 駅からの距離に応じて非線形な重みを付けて合算
    tmp = (
        tmp
        .with_columns(
            (pl.col('distance_from_station_1') // 80).cast(pl.Int64).alias('walk_from_station1'),
            (pl.col('distance_from_station_2') // 80).cast(pl.Int64).alias('walk_from_station2'),
            (pl.col('distance_from_station_3') // 80).cast(pl.Int64).alias('walk_from_station3'),
            )
        .with_columns(
            pl.when(pl.col('station_group_3').is_not_null())
            .then((pl.col('sg1_count2022') * (1 / (1 + np.exp(k * (pl.col('walk_from_station1') - c))))
            + pl.col('sg2_count2022') * (1 / (1 + np.exp(k * (pl.col('walk_from_station2') - c))))
            + pl.col('sg3_count2022') * (1 / (1 + np.exp(k * (pl.col('walk_from_station3') - c))))))

            .when(pl.col('station_group_2').is_not_null())
            .then((pl.col('sg1_count2022') * (1 / (1 + np.exp(k * (pl.col('walk_from_station1') - c))))
            + pl.col('sg2_count2022') * (1 / (1 + np.exp(k * (pl.col('walk_from_station2') - c))))))

            .otherwise(pl.col('sg1_count2022') * (1 / (1 + np.exp(k * (pl.col('walk_from_station1') - c)))))
            .cast(pl.Int64)
            .alias('station_num_weight')
            )
    )
    # 元のデータフレームに結合
    output_df = (output_df.join(tmp.select('building_id', 'station_num_weight'), on='building_id', how='left'))

    return output_df

def get_land_price(input_df, land_kuni_path, land_ken_path):
    
    #土地価格データ読み込み
    land_kuni = gpd.read_file(land_kuni_path)
    land_to = gpd.read_file(land_ken_path)
    
    #必要な部分に結合
    land_kuni = land_kuni[['L01_006', 'geometry']]
    land_kuni.columns = ['price', 'geometry']
    land_to = land_to[['L02_006', 'geometry']]
    land_to.columns = ['price', 'geometry']
    land_df = pd.concat([land_kuni, land_to])
    
    input_land = (
        input_df
        .select('lon', 'lat', 'building_id')
        .unique()
        .to_pandas()
    )
    
    # input_dfをGeoDataFrameに変換
    input_land['geometry'] = input_land.apply(lambda row:Point(row['lon'], row['lat']), axis=1)
    input_land_gdf = gpd.GeoDataFrame(input_land, geometry='geometry', crs='EPSG:4326')
    
    #測地系を変更
    land_df = land_df.to_crs('EPSG:32654')

    #地価公示地点から半径1kmの円状ポリゴンを作成
    radius_km = 1
    land_df['buffer'] = land_df['geometry'].buffer(radius_km * 1000)

    #測地系を戻す
    land_df['geometry'] = land_df['geometry'].to_crs('EPSG:4326')
    land_df['buffer'] = land_df['buffer'].to_crs('EPSG:4326')

    #land_dfに結合用の一意のidを付与
    land_df['land_id'] = range(1, len(land_df.index) + 1)

    # ポリゴンのデータフレームを作成
    land_price_polygon = land_df[['land_id', 'buffer']]
    land_price_polygon = land_price_polygon.rename(columns={'buffer':'geometry'})
    
    # 空間結合
    joined = gpd.sjoin(input_land_gdf, land_price_polygon, how='left', predicate='within')
    
    land_joined = pd.merge(joined, land_df[['land_id', 'price', 'geometry']].rename(columns={'geometry':'land_price_geometry'}), how='left', on='land_id')

    # 距離計算を行う
    land_joined['geometry'] = land_joined['geometry'].to_crs(epsg=3857)
    land_joined['land_price_geometry'] = land_joined['land_price_geometry'].to_crs(epsg=3857)

    land_joined['distance_price_point'] = land_joined['geometry'].distance(land_joined['land_price_geometry'])
    land_joined = land_joined[['building_id', 'land_id', 'distance_price_point', 'price']]

    land_joined['distance_price_point'] = land_joined['distance_price_point'].dropna().round().astype('Int64')
    
    land_sorted = land_joined.sort_values(['building_id', 'distance_price_point'])
    
    # 各building_idごとに距離の順位をつける
    land_sorted['rank'] = land_sorted.groupby('building_id').cumcount() + 1
    
    # 1km圏内の基準点数を計測
    max_count_land_price = land_sorted.groupby('building_id')['rank'].max().reset_index()
    max_count_land_price.rename(columns={'rank':'count_landprice_point'}, inplace=True)

    # 上位3件に絞り込み
    land_sorted = land_sorted[land_sorted['rank'] <= 3]
    
    # 横のカラムに変換
    land_distance = land_sorted.pivot(index='building_id', columns='rank', values=['land_id', 'distance_price_point', 'price'])
    land_distance.columns = [f'{col}_{int(rank)}' for col, rank in land_distance.columns]
    land_distance.reset_index(inplace=True)
    
    # データフレームを結合
    output_df = land_distance.merge(max_count_land_price, on='building_id', how='left') 
    output_df = pl.from_pandas(output_df)
    output_df = input_df.join(output_df, on='building_id', how='left')
    
    return output_df

def make_reform_flg(input_df:pl.DataFrame):
    reform_col = [
        'reform_exterior',
        'reform_exterior_other',
        'reform_exterior_date',
        'reform_common_area',
        'reform_common_area_date',
        'reform_date',
        'reform_place',
        'reform_place_other',
        'reform_wet_area',
        'reform_wet_area_other',
        'reform_wet_area_date',
        'reform_interior',
        'reform_interior_other',
        'reform_interior_date',
        'reform_etc',
        'renovation_date',
        'renovation_etc',
    ]
    
    output_df = input_df.clone()
    
    output_df= (
        output_df
        .with_columns(
            # 外装リフォームのフラグ
            pl.when(pl.col('reform_exterior_date').is_not_null())
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias('reform_exterior_flg'),
            # 共用エリアリフォームのフラグ
            pl.when(pl.col('reform_common_area_date').is_not_null())
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias('reform_common_area_flg'),
            # リフォームのフラグ
            pl.when(pl.col('reform_date').is_not_null())
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias('reform_flg'),
            # 水回りリフォームのフラグ
            pl.when(pl.col('reform_wet_area_date').is_not_null())
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias('reform_wet_area_flg'),
            # 水回りリフォームのフラグ
            pl.when(pl.col('reform_interior_date').is_not_null())
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias('reform_interior_flg'),
            # リノベーションリフォームのフラグ
            pl.when(pl.col('renovation_date').is_not_null())
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias('renovation_flg'),
        )
        .select(pl.all().exclude(reform_col))
    )
    return output_df